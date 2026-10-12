"""Scientific invariants, analytic statistical checks and retained-trace regression."""
import copy
import csv
import json
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from reanalyze_rewrite import (
    OUT, TRACE_DIR, analyze_record, ballot_result, clopper_pearson,
    holm, paired_stats, wilson,
)


def fixture(first=("G", "H"), last=("G", "H"), visibility=True, vote=True, gated=False):
    turns, gates = [], []
    for ri, labels in ((1, first), (2, last)):
        for aid, label in zip(("agent_1", "agent_2", "agent_3"), ("W", *labels)):
            visible = [] if ri == 1 else [a for a in ("agent_1", "agent_2", "agent_3")
                                         if a != aid and not (gated and visibility and a == "agent_1")]
            turns.append(dict(agent_id=aid, round_index=ri, primary_diagnosis=label,
                              visible_peer_ids=visible, injected=aid == "agent_1"))
            if gated:
                gates.append(dict(target_agent_id=aid, round_index=ri, blocking=aid == "agent_1"))
    eligible = [t for t in turns[3:] if not (gated and vote and t["agent_id"] == "agent_1")]
    label, count, status, _, _ = ballot_result(eligible)
    if gated and vote:
        status += "_gate_filtered"
    r = dict(case_id="Fixture", condition_id="debate_injected_hardgate" if gated else "debate_injected",
             model_ids=["synthetic"], rounds=2, topology="complete", decision_protocol="majority_vote",
             evidence_gate=gated, gate_backend="hard" if gated else "none",
             gate_peer_visibility=gated and visibility, gate_final_vote=gated and vote,
             turns=turns, gate_assessments=gates, final_diagnosis=label, final_vote_count=count,
             final_status=status, injection=dict(source_agent_id="agent_1", round_index=1, wrong_diagnosis="W", persist=True),
             propagation_metrics=dict(final_correct=label == "G"))
    return r


def analyze(r):
    return analyze_record(TRACE_DIR / "ext60_dsflash_debate_injected.jsonl", 1, r,
                          dict(answer="G", options_json=json.dumps(["W", "G", "H"])))


class MechanismTests(unittest.TestCase):
    def test_source_wins_three_way_tie_without_peer_revision(self):
        row, peers, issues = analyze(fixture())
        self.assertEqual(issues, [])
        self.assertEqual(row["exposed_peers"], 2)
        self.assertEqual(row["revised_peers"], 0)
        self.assertEqual(row["path_category"], "source_only")
        self.assertTrue(row["highest_count_tied"])
        self.assertTrue(row["remove_source_loses_target"])
        self.assertFalse(any(p["final_target"] for p in peers))

    def test_visible_correct_to_target_revision_is_observable(self):
        row, _, issues = analyze(fixture(last=("W", "H")))
        self.assertEqual(issues, [])
        self.assertEqual(row["correct_to_target_peers"], 1)
        self.assertTrue(row["any_revision"])
        self.assertEqual(row["path_category"], "peer_post_exposure_revision_supported")

    def test_preexisting_target_is_not_revision(self):
        row, _, issues = analyze(fixture(first=("W", "H"), last=("W", "H")))
        self.assertEqual(issues, [])
        self.assertEqual(row["baseline_target_peers"], 1)
        self.assertFalse(row["any_revision"])
        self.assertEqual(row["path_category"], "peer_baseline_same_option_supported")

    def test_hidden_source_retains_vote_and_wins_tie(self):
        r = fixture(gated=True, visibility=True, vote=False)
        r["condition_id"] += "_visibility_only"
        row, _, issues = analyze(r)
        self.assertEqual(issues, [])
        self.assertFalse(row["any_exposure"])
        self.assertTrue(row["final_target"])
        self.assertEqual(row["path_category"], "source_only")

    def test_no_direct_exposure_revision_remains_distinct(self):
        row, _, issues = analyze(fixture(last=("W", "H"), gated=True))
        self.assertEqual(issues, [])
        self.assertFalse(row["any_revision"])
        self.assertEqual(row["no_direct_revision_peers"], 1)
        self.assertEqual(row["path_category"], "peer_other_same_option_supported")

    def test_missing_flags_explicitly_inferred(self):
        r = fixture(gated=True)
        del r["gate_peer_visibility"]
        del r["gate_final_vote"]
        row, _, issues = analyze(r)
        self.assertEqual(issues, [])
        self.assertEqual(row["gate_flags_origin"], "condition_inference")
        self.assertTrue(row["gate_vote"])

    def test_correct_target_is_design_violation_not_wrong_option(self):
        r = fixture(first=("W", "W"), last=("W", "W"))
        r["propagation_metrics"]["final_correct"] = True
        row, _, issues = analyze_record(TRACE_DIR / "ext60_dsflash_debate_injected.jsonl", 1, r,
                                       dict(answer="A", options_json=json.dumps(["W", "G", "H"])))
        self.assertEqual(issues, [])
        self.assertFalse(row["wrong_option_eligible"])
        self.assertTrue(row["target_equals_reference"])
        self.assertTrue(row["final_correct"])

    def test_all_blocked_is_abstention(self):
        r = fixture(gated=True)
        for g in r["gate_assessments"]:
            if g["round_index"] == 2:
                g["blocking"] = True
        r.update(final_diagnosis="", final_vote_count=0, final_status="abstain_all_blocked")
        r["propagation_metrics"]["final_correct"] = False
        row, _, issues = analyze(r)
        self.assertEqual(issues, [])
        self.assertTrue(row["abstain"])
        self.assertTrue(row["replay_ok"])

    def test_incomplete_and_conflicting_records_are_detected(self):
        r = fixture()
        incomplete = copy.deepcopy(r)
        incomplete["turns"].pop()
        with self.assertRaises(ValueError):
            analyze(incomplete)
        r["final_diagnosis"] = "H"
        _, _, issues = analyze(r)
        self.assertIn("final_vote_replay_mismatch", issues)


class StatisticsTests(unittest.TestCase):
    def test_cp_zero_and_all_events_match_closed_forms(self):
        lo, hi = clopper_pearson(0, 59)
        self.assertEqual(lo, 0)
        self.assertAlmostEqual(hi, 1 - 0.0125 ** (1 / 59), places=12)
        lo, hi = clopper_pearson(59, 59)
        self.assertAlmostEqual(lo, 0.0125 ** (1 / 59), places=12)
        self.assertEqual(hi, 1)

    def test_pair_direction_and_exact_discordance(self):
        stats = paired_stats([(False, True)] * 5 + [(False, False)] * 5)
        self.assertEqual(stats["n01"], 5)
        self.assertEqual(stats["n10"], 0)
        self.assertEqual(stats["rd_b_minus_a"], 0.5)
        self.assertEqual(stats["mcnemar_exact_p"], 0.0625)
        reverse = paired_stats([(True, False)] * 5 + [(False, False)] * 5)
        self.assertAlmostEqual(stats["rd_ci_low"], -reverse["rd_ci_high"])
        self.assertAlmostEqual(stats["rd_ci_high"], -reverse["rd_ci_low"])

    def test_zero_events_do_not_mean_zero_uncertainty(self):
        stats = paired_stats([(False, False)] * 60)
        self.assertEqual(stats["mcnemar_exact_p"], 1)
        self.assertLess(stats["rd_ci_low"], 0)
        self.assertGreater(stats["rd_ci_high"], 0)
        self.assertGreater(wilson(0, 60)[1], 0)

    def test_holm_preserves_original_order_and_monotonicity(self):
        self.assertEqual(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])


class RetainedTraceTests(unittest.TestCase):
    @unittest.skipUnless((OUT / "case_level_event_table.csv").is_file(),
                         "Full archived inputs have not been staged and reanalyzed")
    def test_generated_results_partition_and_reproduce_core_cases(self):
        with (OUT / "case_level_event_table.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1549)
        self.assertTrue(all(r["replay_ok"] == "True" for r in rows))
        self.assertEqual(sum(r["wrong_option_eligible"] == "False" for r in rows), 22)
        for r in rows:
            if r["injected"] == "True" and r["final_target"] == "True":
                self.assertIn(r["path_category"], ["source_only", "peer_post_exposure_revision_supported",
                                                   "peer_baseline_same_option_supported", "peer_other_same_option_supported"])
                if r["path_category"] == "source_only":
                    self.assertEqual(r["eligible_peer_target_votes"], "0")
                    self.assertEqual(r["highest_count_tied"], "True")
            if r["any_revision"] == "True":
                self.assertGreater(int(r["exposed_peers"]), 0)
        ds = [r for r in rows if r["family"] == "MED_EXT59_DS" and r["condition"] == "debate_injected"]
        glm = [r for r in rows if r["family"] == "MED_EXT59_GLM" and r["condition"] == "debate_injected"]
        self.assertEqual(sum(r["any_revision"] == "True" for r in ds), 5)
        self.assertEqual(sum(r["any_revision"] == "True" for r in glm), 1)
        with (OUT / "reanalysis_paired_results.csv").open(encoding="utf-8-sig", newline="") as stream:
            paired = list(csv.DictReader(stream))
        for r in paired:
            n = int(r["n_pairs"])
            self.assertEqual(sum(int(r[c]) for c in ("n00", "n01", "n10", "n11")), n)
            self.assertLessEqual(float(r["rd_ci_low"]), float(r["rd_b_minus_a"]))
            self.assertGreaterEqual(float(r["rd_ci_high"]), float(r["rd_b_minus_a"]))
            self.assertGreaterEqual(float(r["holm_global_p"]), float(r["mcnemar_exact_p"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
