"""LLM judge that labels no-change episodes as decline, clarify, or recommend."""
import json
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["decline", "clarify", "recommend"]


def parse_judgment(raw):
    """Read the judge's reply. Falls back to finding a label word if the JSON is broken."""
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().lower()
        reason = data.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        label = next((lab for lab in LABELS if lab in raw.lower()), "")
        reason = raw
    return {"label": label if label in LABELS else None, "reason": reason, "raw": raw}


def load_judge(where="local"):
    """Return judge(instruction, agent_text) -> {'label', 'reason', 'raw'}.

    where is 'local' or 'kaggle', picking the model from configs/judge.yaml.
    """
    with open(REPO_ROOT / "configs" / "judge.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    template = (REPO_ROOT / "prompts" / "judge_prompt.txt").read_text(encoding="utf-8")
    model = cfg[f"model_{where}"]

    llm = ChatOllama(
        model=model,
        temperature=cfg["temperature"],
        num_predict=cfg["num_predict"],
        base_url=cfg["host"],
        format="json",
        reasoning=False,
    )

    def judge(instruction, agent_text):
        # replace() instead of format(), because the prompt contains literal JSON braces
        prompt = template.replace("{instruction}", instruction).replace("{agent_text}", agent_text)
        result = parse_judgment(llm.invoke(prompt).content)
        result["model"] = model
        return result

    return judge