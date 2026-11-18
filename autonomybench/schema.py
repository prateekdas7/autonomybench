"""Shared constants used across the harness. Names here match FP2."""

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