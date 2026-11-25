"""Agent loop for AutonomyBench.

Implements the LangGraph StateGraph.

Four nodes run in sequence for every episode:
  load_scenario  — builds a fresh World from the scenario JSON.
  agent          — calls the model with the system prompt, conversation history,
                   and all tool schemas, returns a tool call or a final reply.
  tool_executor  — runs the tool which the model requested, appends the result to the
                   conversation, and loops back to agent.
  end_episode    — records why the episode ended and saves the trajectory.

Usage:

    from autonomybench.graph import build_graph, run_episode
    from autonomybench.world import load_scenario

    graph = build_graph(model_tag, cfg)
    scenario = load_scenario(scenario_id)
    traj  = run_episode(graph, scenario, trial)
"""

import inspect
import functools
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END

from autonomybench.schema import TOOL_KIND, make_trajectory
from autonomybench.tools import TOOL_FN
from autonomybench.world import World


# Graph State
class EpisodeState(MessagesState):
    """State shared across all nodes during one episode.

    MessagesState gives us messages with the add_messages reducer, which
    appends new messages rather than replacing the whole list.

    We add three more fields on top of that:

    world      — the live World object for this episode.
    step_count — incremented by tool_executor on every tool call. The router
                 checks it against the cap before handing control back to agent.
    end_reason — set by end_episode so the caller knows why the loop stopped.
    """
    world: object
    step_count: int
    end_reason: str
    config: dict


# Prompt
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_TEMPLATE = (_REPO_ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")


def _build_system_message(scenario: dict) -> SystemMessage:
    """Fills user_name, user_email, and today's date into system prompt template.

    Returns a SystemMessage to add at the beginning of every model call.
    """
    user = scenario["user"]
    text = (
        _PROMPT_TEMPLATE
        .replace("{user_name}",  user["name"])
        .replace("{user_email}", user["email"])
        .replace("{today}",      scenario["today"])
    )
    return SystemMessage(content=text)


# Tool Wrappers
def _make_lc_tools(world_ref: list) -> list:
    """Wrap every function in TOOL_FN as a LangChain StructuredTool.

    world_ref is a one-element list. Updating world_ref[0] before each episode
    points all tools at the new World without rebuilding anything.

    For each tool, we remove 'world' from the original signature and stamp the
    trimmed version onto the wrapper. 

    LangChain reads __signature__ to build the JSON schema it sends to the model, 
    so that the model only sees the parameters it needs to supply.
    """
    lc_tools = []

    for name, fn in TOOL_FN.items():

        def _make_tool(fn=fn, name=name):
            # Remove 'world' so that the model does not see it in the schema
            orig_sig = inspect.signature(fn)
            model_params = [p for p in orig_sig.parameters.values() if p.name != "world"]
            trimmed_sig = orig_sig.replace(parameters=model_params)

            # Inject the live World at call time
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(world_ref[0], *args, **kwargs)

            wrapper.__signature__ = trimmed_sig

            return StructuredTool.from_function(
                func=wrapper,
                name=name,
                description=(fn.__doc__ or name).strip(),
            )

        lc_tools.append(_make_tool())

    return lc_tools


# Graph Nodes
def _load_scenario_node(state: EpisodeState) -> dict:
    """First node after START. The World is already in state from run_episode().

    Returns an empty dict. 

    Exists as a named node to mark the clean start of each episode in the graph trace.
    """
    return {}


def _route(state: EpisodeState) -> str:
    """Conditional edge that runs after every agent call.

    Returns the name of the next node. Checks in this order:

      1. Step cap reached                    : end_episode
      2. No tool call in reply               : end_episode  (final answer)
      3. ask_user or request_approval called : end_episode
      4. Any other tool called               : tool_executor
    """
    cap = state.get("config", {}).get("max_tool_calls", 15)

    if state["step_count"] >= cap:
        return "end_episode"

    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end_episode"

    if last.tool_calls[0]["name"] in ("ask_user", "request_approval"):
        return "end_episode"

    return "tool_executor"


def _make_tool_executor_node(tool_map: dict, world_ref: list):
    """Return the tool_executor node capturing tool_map and world_ref.

    Runs every tool call in the last AIMessage, builds a ToolMessage for each
    result so the model sees it on the next step, and increments step_count.

    If the model names a tool that does not exist, logs the error and returns
    an error string rather than crashing.
    """
    def _tool_executor_node(state: EpisodeState) -> dict:
        last = state["messages"][-1]
        tool_messages = []

        for call in last.tool_calls:
            name = call["name"]
            args = call["args"]
            tool_obj = tool_map.get(name)

            if tool_obj is None:
                # Log the error
                err = f"Error: unknown tool '{name}'."
                world_ref[0].record(name, TOOL_KIND.get(name, "read"), args, {"error": err})
                result = err
            else:
                # invoke() runs the wrapper which calls tools.py and records to world.log
                result = tool_obj.invoke(args)

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        return {
            "messages": tool_messages,
            "step_count": state["step_count"] + 1,
        }

    return _tool_executor_node


def _make_end_episode_node(world_ref: list, scenario_ref: list, traj_store: list):
    """Return the end_episode node capturing world_ref, scenario_ref, and traj_store.

    Figures out why the episode ended, pulls the final message text, calls
    make_trajectory() to build the full record, and stores it in traj_store[0],
    so that run_episode() can return it to the caller.
    """
    def _end_episode_node(state: EpisodeState) -> dict:
        world = world_ref[0]
        scenario = scenario_ref[0]
        cfg = state.get("config", {})
        cap = cfg.get("max_tool_calls", 15)

        messages = state["messages"]
        last = messages[-1] if messages else None

        # Determine end reason
        if state["step_count"] >= cap:
            reason = "step_limit"
        elif last and hasattr(last, "tool_calls") and last.tool_calls:
            tool_name = last.tool_calls[0]["name"]
            if tool_name == "ask_user":
                reason = "ask_user"
            elif tool_name == "request_approval":
                reason = "request_approval"
            else:
                reason = "final_answer"
        else:
            reason = "final_answer"

        # Extract the text the agent gave to the user
        final_text = ""
        if last:
            if hasattr(last, "content") and isinstance(last.content, str):
                final_text = last.content
            elif hasattr(last, "tool_calls") and last.tool_calls:
                final_text = str(last.tool_calls[0].get("args", {}))

        traj_store[0] = make_trajectory(
            world=world,
            scenario=scenario,
            model=cfg.get("model", "unknown"),
            trial=cfg.get("trial", 1),
            end_reason=reason,
            final_message=final_text,
            config=cfg,
        )

        return {"end_reason": reason}

    return _end_episode_node


# Graph Builder
def build_graph(model_tag: str, cfg: dict | None = None):
    """Compile and return the AutonomyBench graph for one model.

    Call this once per model at the start of a run. 

    The returned graph is reused across all scenarios and trials for that 
    model - run_episode() updates the World and scenario before each invoke().
    """
    cfg = cfg or {}

    # Shared mutable containers
    world_ref = [None]
    scenario_ref = [None]
    traj_store = [None]
    sys_msg_ref = [SystemMessage(content="")]

    # Build the model
    llm = ChatOllama(
        model=model_tag,
        temperature=cfg.get("temperature", 0.7),
        num_predict=cfg.get("num_predict", 2048),
        base_url=cfg.get("ollama_host", "http://localhost:11434"),
        reasoning=False,
        client_kwargs={"timeout": 300},
    )

    # Wrap tools and bind to model
    lc_tools = _make_lc_tools(world_ref)
    tool_map = {t.name: t for t in lc_tools}
    model_with_tools = llm.bind_tools(lc_tools)

    # Agent node: adds the system message at the beginning of every call
    def _agent_node(state: EpisodeState) -> dict:
        reply = model_with_tools.invoke([sys_msg_ref[0]] + state["messages"])
        return {"messages": [reply]}

    # Construct the state graph
    builder = StateGraph(EpisodeState)

    builder.add_node("load_scenario", _load_scenario_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tool_executor", _make_tool_executor_node(tool_map, world_ref))
    builder.add_node("end_episode", _make_end_episode_node(world_ref, scenario_ref, traj_store))

    builder.add_edge(START, "load_scenario")
    builder.add_edge("load_scenario", "agent")
    builder.add_edge("tool_executor", "agent")
    builder.add_edge("end_episode", END)

    builder.add_conditional_edges("agent", _route, ["tool_executor", "end_episode"])

    graph = builder.compile()

    # Attach refs, so that run_episode() can reach them without global variables
    graph._world_ref = world_ref
    graph._scenario_ref = scenario_ref
    graph._traj_store = traj_store
    graph._sys_msg_ref = sys_msg_ref
    graph._cfg = cfg
    graph._model_tag = model_tag

    return graph


# Episode Runner
def run_episode(graph, scenario: dict, trial: int = 1) -> dict:
    """Run one scenario through the graph and return the finished trajectory.

    Resets all shared state before invoking, so that each episode starts clean.

    The trial number is recorded in the trajectory for consistency scoring.

    Returns the trajectory dict produced by make_trajectory().
    """
    cfg = dict(graph._cfg)
    cfg["model"] = graph._model_tag
    cfg["trial"] = trial

    # Fresh World for this episode
    world = World(scenario)
    graph._world_ref[0] = world
    graph._scenario_ref[0] = scenario
    graph._traj_store[0] = None
    graph._sys_msg_ref[0] = _build_system_message(scenario)

    graph.invoke({
        "messages": [HumanMessage(content=scenario["instruction"])],
        "world": world,
        "step_count": 0,
        "end_reason": "",
        "config": cfg,
    })

    return graph._traj_store[0]


# Config Loader
def load_config(profile: str = "kaggle") -> dict:
    """Read configs/conditions.yaml and return a settings dict.
    """
    raw = yaml.safe_load((_REPO_ROOT / "configs" / "conditions.yaml").read_text(encoding="utf-8"))

    return {
        "models": raw.get("agent_models", {}).get(profile, []),
        "temperature": raw.get("sampling", {}).get("temperature", 0.7),
        "num_predict": raw.get("sampling", {}).get("num_predict", 2048),
        "seed": raw.get("sampling", {}).get("seed"),
        "max_tool_calls": raw.get("episode", {}).get("max_tool_calls", 15),
        "trials_per_scenario": raw.get("episode", {}).get("trials_per_scenario", 3),
        "ollama_host": raw.get("ollama", {}).get("host", "http://localhost:11434"),
        "scenarios_split": raw.get("scenarios", {}).get("split", "pilot"),
        "runs_dir": raw.get("output", {}).get("runs_dir", "results/runs"),
    }
