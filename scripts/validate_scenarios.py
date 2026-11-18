"""Check every scenario in scenarios/<split>/ and every A/B pair.

Usage: python scripts/validate_scenarios.py pilot
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autonomybench.schema import validate_scenario, validate_pair  # noqa: E402


def main(split="pilot"):
    files = sorted((REPO_ROOT / "scenarios" / split).glob("*.json"))
    if not files:
        print(f"No scenario files in scenarios/{split}/")
        return 1

    pairs = defaultdict(dict)
    problems = 0

    for f in files:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"{f.name}: invalid JSON ({e})")
            problems += 1
            continue
        errors = validate_scenario(s)
        if errors:
            problems += 1
            print(f"{f.name}:")
            for e in errors:
                print(f"  - {e}")
        if "pair_id" in s and "variant" in s:
            pairs[s["pair_id"]][s["variant"]] = s

    print()
    for pair_id, variants in sorted(pairs.items()):
        if set(variants) != {"A", "B"}:
            print(f"{pair_id}: missing a variant (have {sorted(variants)})")
            problems += 1
            continue
        errors, diffs = validate_pair(variants["A"], variants["B"])
        if errors:
            problems += 1
            print(f"{pair_id}:")
            for e in errors:
                print(f"  - {e}")
        print(f"{pair_id}: world differs at {diffs}")

    print(f"\n{len(files)} files, {len(pairs)} pairs, {problems} with problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pilot"))