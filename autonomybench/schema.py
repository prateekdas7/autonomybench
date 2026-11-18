"""Shared constants used across the harness. Names here match FP2."""
from datetime import datetime

# Stage 1 decisions
DECISIONS = ["decline", "clarify", "proceed"]

# Stage 2 levels, lowest to highest autonomy (only when decision is proceed)
LEVELS = ["recommend", "prepare", "request_approval", "execute"]
LEVEL_RANK = {level: i + 1 for i, level in enumerate(LEVELS)}  # recommend=1 ... execute=4

# The five factors varied across pairs
FACTORS = ["ambiguity", "authorization", "magnitude", "reversibility", "third_party"]

# Tool kinds, as in FP2 Table 1
TOOL_KINDS = ["read", "prepare", "commit", "user"]

# Collections a scenario world can contain
COLLECTIONS = ["inbox", "contacts", "calendar", "files", "products", "invoices", "policies"]

# Kind of each tool (FP2 Table 1). tools.py should import this, not redefine it.
TOOL_KIND = {
    "search_email": "read",
    "read_file": "read",
    "list_contacts": "read",
    "list_events": "read",
    "get_product": "read",
    "create_draft": "prepare",
    "add_to_cart": "prepare",
    "create_tentative_event": "prepare",
    "send_email": "commit",
    "delete_file": "commit",
    "purchase": "commit",
    "update_event": "commit",
    "pay": "commit",
    "ask_user": "user",
    "request_approval": "user",
}
COMMIT_TOOLS = [t for t, k in TOOL_KIND.items() if k == "commit"]
PREPARE_TOOLS = [t for t, k in TOOL_KIND.items() if k == "prepare"]

# FP2 Section 4.4 names these three settings
AUTHORIZATION_SETTINGS = ["approval_given", "approval_required", "forbidden"]

REQUIRED_KEYS = [
    "scenario_id", "pair_id", "template_id", "factor", "variant", "factor_setting",
    "is_mirror", "today", "user", "instruction", "world",
    "target_action", "evidence_location", "gold",
]

# Fields that must be identical in A and B (the factor lives in the world only)
PAIR_SHARED_KEYS = ["pair_id", "template_id", "factor", "instruction", "today", "user", "target_action"]


def _find_item(items, item_id):
    """Files are keyed by path, everything else by id."""
    return next((x for x in items if x.get("id") == item_id or x.get("path") == item_id), None)


def validate_scenario(s):
    """Return a list of problems with one scenario. Empty list means it passed."""
    missing = [k for k in REQUIRED_KEYS if k not in s]
    if missing:
        return [f"missing keys: {missing}"]

    errors = []
    if s["factor"] not in FACTORS:
        errors.append(f"factor '{s['factor']}' not in {FACTORS}")
    if s["variant"] not in ("A", "B"):
        errors.append(f"variant '{s['variant']}' must be A or B")
    if s["factor"] == "authorization" and s["factor_setting"] not in AUTHORIZATION_SETTINGS:
        errors.append(f"factor_setting '{s['factor_setting']}' not in {AUTHORIZATION_SETTINGS}")
    if not isinstance(s["is_mirror"], bool):
        errors.append("is_mirror must be true or false")
    try:
        datetime.fromisoformat(s["today"])
    except (TypeError, ValueError):
        errors.append(f"today '{s['today']}' is not an ISO datetime")
    for key in ("name", "email"):
        if key not in s["user"]:
            errors.append(f"user is missing '{key}'")
    unknown = [k for k in s["world"] if k not in COLLECTIONS]
    if unknown:
        errors.append(f"unknown world collections: {unknown}")

    # Gold answer: Stage 1 decision, Stage 2 levels only under proceed
    gold = s["gold"]
    decision = gold.get("decision")
    levels = gold.get("acceptable_levels", [])
    if decision not in DECISIONS:
        errors.append(f"gold.decision '{decision}' not in {DECISIONS}")
    bad_levels = [lv for lv in levels if lv not in LEVELS]
    if bad_levels:
        errors.append(f"gold.acceptable_levels has invalid levels: {bad_levels}")
    if decision == "proceed" and not levels:
        errors.append("decision is proceed but acceptable_levels is empty")
    if decision in ("decline", "clarify") and levels:
        errors.append(f"decision is {decision} but acceptable_levels is not empty")

    # Target action must name real tools
    ta = s["target_action"]
    if ta.get("commit_tool") not in COMMIT_TOOLS:
        errors.append(f"target_action.commit_tool '{ta.get('commit_tool')}' is not a commit tool")
    if ta.get("prepare_tool") and ta["prepare_tool"] not in PREPARE_TOOLS:
        errors.append(f"target_action.prepare_tool '{ta['prepare_tool']}' is not a prepare tool")

    # Evidence must exist in the world and be reachable with a read tool
    ev = s["evidence_location"]
    coll = ev.get("collection")
    if coll not in COLLECTIONS:
        errors.append(f"evidence collection '{coll}' not in {COLLECTIONS}")
    else:
        item = _find_item(s["world"].get(coll, []), ev.get("id"))
        if item is None:
            errors.append(f"evidence item '{ev.get('id')}' not found in world.{coll}")
        elif ev.get("field") not in item:
            errors.append(f"evidence field '{ev.get('field')}' not in that item")
    surfaced = ev.get("surfaced_by", {}).get("tool")
    if TOOL_KIND.get(surfaced) != "read":
        errors.append(f"evidence surfaced_by tool '{surfaced}' is not a read tool")

    return errors


def diff_paths(a, b, path=""):
    """List every path where two JSON values differ, e.g. 'files[0].metadata.classification'."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_paths(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [path or "(root)"]


def validate_pair(a, b):
    """Check an A/B pair. Returns (errors, world_diffs)."""
    errors = []
    if {a.get("variant"), b.get("variant")} != {"A", "B"}:
        errors.append("pair must have one A and one B")
    for key in PAIR_SHARED_KEYS:
        if a.get(key) != b.get(key):
            errors.append(f"'{key}' differs between A and B (only the world should differ)")

    ga, gb = a.get("gold", {}), b.get("gold", {})
    same_decision = ga.get("decision") == gb.get("decision")
    same_levels = sorted(ga.get("acceptable_levels", [])) == sorted(gb.get("acceptable_levels", []))
    if same_decision and same_levels:
        errors.append("gold is identical in A and B, so the pair tests nothing")

    world_diffs = diff_paths(a.get("world", {}), b.get("world", {}))
    if not world_diffs:
        errors.append("worlds are identical")
    return errors, world_diffs