"""Synthetic illustration of trace categories; no external inputs or model calls."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_reanalyze_rewrite import analyze, fixture


def main():
    examples = {
        "source_only_tie": fixture(),
        "post_exposure_revision": fixture(last=("W", "H")),
        "baseline_agreement": fixture(first=("W", "H"), last=("W", "H")),
        "source_hidden_but_still_voting": fixture(gated=True, visibility=True, vote=False),
    }
    examples["source_hidden_but_still_voting"]["condition_id"] += "_visibility_only"
    for name, record in examples.items():
        row, _, issues = analyze(record)
        if issues:
            raise RuntimeError(issues)
        fields = ("path_category", "final_target", "any_exposure", "any_revision", "highest_count_tied")
        print(json.dumps(dict(example=name, **{field: row[field] for field in fields})))


if __name__ == "__main__":
    main()
