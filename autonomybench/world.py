"""Mock world for one AutonomyBench episode."""
import copy
import json
from pathlib import Path

from autonomybench.schema import COLLECTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "scenarios"


def load_scenario(scenario_id, split="pilot"):
    """Read a scenario JSON by id, e.g. load_scenario('P01_A').

    split is 'pilot' or 'full', matching the folders under scenarios/.
    """
    path = SCENARIO_DIR / split / f"{scenario_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class World:
    """A fresh, mutable copy of one scenario's world.

    Collections come from the scenario file. Action lists start empty
    and fill up as the agent prepares or commits things.
    """

    def __init__(self, scenario):
        self.scenario_id = scenario["scenario_id"]
        self.today = scenario["today"]
        self.user = scenario["user"]

        world = copy.deepcopy(scenario["world"])  # never touch the original
        for name in COLLECTIONS:
            setattr(self, name, world.get(name, []))

        # Created by prepare tools
        self.drafts = []
        self.carts = []
        self.tentative_events = []

        # Created by commit tools
        self.sent = []
        self.orders = []
        self.payments = []
        self.event_updates = []
        self.deleted_files = []

        self.log = []  # every tool call, in order
        self._next_id = 0

    def new_id(self, prefix):
        """Sequential ids like 'draft_001', so runs are reproducible."""
        self._next_id += 1
        return f"{prefix}_{self._next_id:03d}"

    def record(self, tool, kind, args, result):
        """Append one tool call to the log. kind is read / prepare / commit / user."""
        entry = {
            "step": len(self.log) + 1,
            "tool": tool,
            "kind": kind,
            "args": args,
            "result": result,
        }
        self.log.append(entry)
        return entry

    def actions(self):
        """Everything the agent changed, for scoring and the saved trajectory."""
        return {
            "drafts": self.drafts,
            "carts": self.carts,
            "tentative_events": self.tentative_events,
            "sent": self.sent,
            "orders": self.orders,
            "payments": self.payments,
            "event_updates": self.event_updates,
            "deleted_files": self.deleted_files,
        }