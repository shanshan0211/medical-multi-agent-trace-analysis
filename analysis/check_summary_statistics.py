"""Check exported counts/statistics, optionally compare a fresh full reanalysis."""
import argparse
import csv
import math
from pathlib import Path

from reanalyze_rewrite import clopper_pearson, holm, paired_stats, wilson

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError(f"Statistical mismatch: {a} != {b}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-generated", action="store_true")
    args = parser.parse_args()
    arms = rows(ROOT / "results/reanalysis_arm_results.csv")
    pairs = rows(ROOT / "results/reanalysis_paired_results.csv")
    assert len(arms) == 48 and len(pairs) == 76
    for arm in arms:
        for endpoint in ("final_target", "final_correct", "any_revision", "revision_and_final_target", "abstain", "any_exposure"):
            n = int(arm[endpoint + "_n"])
            if n:
                k = int(arm[endpoint + "_events"])
                lo, hi = wilson(k, n)
                near(arm[endpoint + "_rate"], k/n)
                near(arm[endpoint + "_ci_low"], lo)
                near(arm[endpoint + "_ci_high"], hi)
    ps = []
    for pair in pairs:
        outcomes = []
        for a, b, field in ((False, False, "n00"), (False, True, "n01"), (True, False, "n10"), (True, True, "n11")):
            outcomes.extend([(a, b)] * int(pair[field]))
        computed = paired_stats(outcomes)
        for field, value in computed.items():
            if isinstance(value, (int, float)):
                near(pair[field], value)
            else:
                assert pair[field] == value
        assert int(pair["holm_family_size"]) == len(pairs)
        ps.append(computed["mcnemar_exact_p"])
    for pair, value in zip(pairs, holm(ps)):
        near(pair["holm_global_p"], value)
    if args.compare_generated:
        for name, exported in (("reanalysis_arm_results.csv", arms), ("reanalysis_paired_results.csv", pairs)):
            regenerated = rows(ROOT / "paper/rewrite/evidence" / name)
            assert len(exported) == len(regenerated)
            for a, b in zip(exported, regenerated):
                for field, value in a.items():
                    other = b[field]
                    if field == "trace":
                        other = other.removeprefix("local_inputs/")
                    assert value == other, (name, field)
        figure = rows(ROOT / "results/figure2_source_data.csv")
        from build_figures import figure_results
        _, generated_figure = figure_results()
        assert figure == [{k: str(v) for k, v in r.items()} for r in generated_figure]
    print("Passed: 48 arm summaries, 76 paired comparisons, global Holm" +
          (", full regenerated result equality and 12 figure rows." if args.compare_generated else "."))


if __name__ == "__main__":
    main()
