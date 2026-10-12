"""Audit fixed-ballot tie-policy sensitivity for the prioritized ungated arms.

This is a post hoc replay of archived final ballots, not a new model run.
"""

import csv
import json
from collections import Counter
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parents[1] / "paper" / "rewrite" / "evidence"
FAMILIES = ("MED_EXT59_DS", "MED_EXT59_GLM")


def votes_from_json(value):
    ballots = json.loads(value)
    if not ballots:
        raise ValueError("Empty eligible ballot list")
    return [next(iter(ballot.values())) for ballot in ballots]


def replay(votes, policy):
    counts = Counter(votes)
    highest = max(counts.values())
    if policy == "majority":
        return next((choice for choice, count in counts.items() if count > len(votes) / 2), None)
    ordered = votes if policy == "first" else reversed(votes)
    return next(choice for choice in ordered if counts[choice] == highest)


def main():
    with (EVIDENCE / "case_level_event_table.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family and row["condition"] == "debate_injected"]
        counts = {policy: 0 for policy in ("first", "last", "majority")}
        ties = 0
        for row in selected:
            votes = votes_from_json(row["eligible_ballots_json"])
            assert replay(votes, "first") == row["final_choice"], row["case_id"]
            leaders = Counter(votes)
            ties += sum(count == max(leaders.values()) for count in leaders.values()) > 1
            for policy in counts:
                counts[policy] += replay(votes, policy) == row["target"]
        print(f"{family}: n={len(selected)}, tied ballots={ties}, target selections={counts}")


if __name__ == "__main__":
    main()
