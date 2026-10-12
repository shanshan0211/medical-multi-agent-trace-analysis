"""Independent read-only reconstruction of trace events and statistics.

The author-facing clinical packet and manuscript report writers are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
EVIDENCE = ROOT / "paper/rewrite/evidence"
ENDPOINTS = ("final_target", "final_correct", "any_revision", "revision_and_final_target")
CATEGORIES = ("source_only", "peer_post_exposure_revision_supported",
              "peer_baseline_same_option_supported", "peer_other_same_option_supported")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def norm(value):
    return " ".join(re.sub(r"[^\w]+", " ", str(value or "").casefold()).split())


def same(a, b):
    return bool(norm(a)) and norm(a) == norm(b)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check_equal(actual, expected, context):
    if str(actual) != str(expected):
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def near(actual, expected, context):
    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"{context}: {actual} != {expected}")


def decide(ballots):
    if not ballots:
        return "", 0, False, False
    counts = Counter(norm(t["primary_diagnosis"]) for t in ballots)
    maximum = max(counts.values())
    winner = next(t["primary_diagnosis"] for t in ballots
                  if counts[norm(t["primary_diagnosis"])] == maximum)
    return winner, maximum, sum(v == maximum for v in counts.values()) > 1, maximum > len(ballots) / 2


def binomial_cdf(k, n, p):
    # Direct binomial sum, independent of original log-sum implementation.
    return math.fsum(math.comb(n, j) * p ** j * (1-p) ** (n-j) for j in range(k+1))


def quantile_by_cdf(k, n, probability):
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp(k, n):
    alpha = 0.025  # Each marginal interval has 97.5% coverage.
    return (0.0 if k == 0 else quantile_by_cdf(k-1, n, 1-alpha/2),
            1.0 if k == n else quantile_by_cdf(k, n, alpha/2))


def subset_rows(rows, subset):
    if subset == "all_retained":
        return rows
    if subset == "wrong_option_eligible":
        return [r for r in rows if r["wrong_option_eligible"] is True]
    if subset == "injection_design_violation":
        return [r for r in rows if r["injected"] and r["wrong_option_eligible"] is False]
    raise ValueError(subset)


def reconstruct(row, raw, ref):
    options = json.loads(ref["options_json"])
    gold = ref["answer"].strip()
    if len(gold) == 1 and gold.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and gold not in options:
        gold = options[ord(gold.upper())-ord("A")]
    turns = raw["turns"]
    by_agent = defaultdict(dict)
    for t in turns:
        if t["round_index"] in by_agent[t["agent_id"]]:
            raise ValueError("Duplicate round")
        by_agent[t["agent_id"]][t["round_index"]] = t
    for history in by_agent.values():
        check_equal(sorted(history), list(range(1, raw["rounds"]+1)), "round completeness")
    injected = bool(raw.get("injection"))
    injection = raw.get("injection") or {}
    target = injection.get("wrong_diagnosis", row["target"])
    source = injection.get("source_agent_id", "")
    gated = bool(raw.get("evidence_gate"))
    condition = raw["condition_id"]
    visibility = bool(raw.get("gate_peer_visibility", gated and not condition.endswith("_vote_only")))
    vote = bool(raw.get("gate_final_vote", gated and not condition.endswith("_visibility_only")))
    origin = "recorded" if "gate_peer_visibility" in raw and "gate_final_vote" in raw else "condition_inference"
    gates = raw.get("gate_assessments") or []
    if gated:
        check_equal(sorted((g["round_index"], g["target_agent_id"]) for g in gates),
                    sorted((t["round_index"], t["agent_id"]) for t in turns), "gate coverage")
    for t in turns:
        ri = t["round_index"]
        blocked_prior = {g["target_agent_id"] for g in gates if g["round_index"] == ri-1 and g["blocking"]} if visibility else set()
        expected = set(by_agent) - {t["agent_id"]} - blocked_prior if ri > 1 else set()
        check_equal(sorted(t["visible_peer_ids"]), sorted(expected), "message visibility")
        if injected and t["agent_id"] == source:
            active = ri >= injection["round_index"] if injection.get("persist") else ri == injection["round_index"]
            check_equal(bool(t.get("injected")), active, "injection schedule")
            if active:
                check_equal(norm(t["primary_diagnosis"]), norm(target), "injected choice")
    final_turns = [t for t in turns if t["round_index"] == raw["rounds"]]
    blocked = {g["target_agent_id"] for g in gates if g["round_index"] == raw["rounds"] and g["blocking"]} if vote else set()
    ballots = final_turns[:1] if raw["decision_protocol"] == "first_agent" else [t for t in final_turns if t["agent_id"] not in blocked]
    choice, count, tied, majority = decide(ballots)
    status = "first_agent" if raw["decision_protocol"] == "first_agent" else ("majority_vote" if majority else "plurality_tiebreak")
    if not ballots:
        status = "abstain_all_blocked"
    elif blocked and raw["decision_protocol"] != "first_agent":
        status += "_gate_filtered"
    check_equal(norm(choice), norm(raw["final_diagnosis"]), "raw final answer")
    check_equal(count, raw["final_vote_count"], "raw vote count")
    check_equal(status, raw["final_status"], "raw final status")
    peer_rows = []
    eligible_ids = {t["agent_id"] for t in ballots}
    if injected:
        for aid, history in by_agent.items():
            if aid == source:
                continue
            exposed, revised, indirect, correct_revision = [], [], [], []
            for ri, t in sorted(history.items()):
                previous = history.get(ri-1)
                source_previous = by_agent[source].get(ri-1)
                direct = bool(source in t["visible_peer_ids"] and source_previous and source_previous.get("injected") and same(source_previous["primary_diagnosis"], target))
                if direct:
                    exposed.append(ri)
                changed = bool(previous and not same(previous["primary_diagnosis"], target) and same(t["primary_diagnosis"], target))
                if changed:
                    (revised if direct else indirect).append(ri)
                    if direct and same(previous["primary_diagnosis"], gold):
                        correct_revision.append(ri)
            peer_rows.append(dict(agent_id=aid, baseline_choice=history[1]["primary_diagnosis"],
                baseline_no_peers=not history[1]["visible_peer_ids"], baseline_target=same(history[1]["primary_diagnosis"], target),
                direct_exposure_rounds=";".join(map(str, exposed)), post_exposure_revision_rounds=";".join(map(str, revised)),
                no_direct_exposure_revision_rounds=";".join(map(str, indirect)), correct_to_target_revision_rounds=";".join(map(str, correct_revision)),
                exposed=bool(exposed), revised=bool(revised), no_direct_revision=bool(indirect), correct_to_target=bool(correct_revision),
                final_choice=history[raw["rounds"]]["primary_diagnosis"], final_target=same(history[raw["rounds"]]["primary_diagnosis"], target),
                eligible_final_vote=aid in eligible_ids))
    support = [p for p in peer_rows if p["eligible_final_vote"] and p["final_target"]]
    selected = same(choice, target) if target else None
    category = "not_target_selected" if target else "not_applicable"
    if injected and selected:
        category = (CATEGORIES[0] if not support else CATEGORIES[1] if any(p["revised"] for p in support)
                    else CATEGORIES[2] if any(p["baseline_target"] for p in support) else CATEGORIES[3])
    elif not injected and selected:
        category = "matched_target_control"
    removed = decide([t for t in ballots if t["agent_id"] != source])[0]
    result = dict(reference=gold, stored_gold=raw.get("gold_diagnosis", ""), target=target, injected=injected,
        final_choice=choice, final_correct=same(choice, gold), final_target=selected,
        wrong_option_eligible=bool(any(same(target, o) for o in options) and not same(target, gold)) if injected else "",
        target_equals_reference=same(target, gold) if target else "", gate_visibility=visibility, gate_vote=vote, gate_flags_origin=origin,
        abstain=not bool(choice.strip()), final_outside_options=bool(choice) and not any(same(choice, o) for o in options),
        any_exposure=any(p["exposed"] for p in peer_rows) if injected else None,
        any_revision=any(p["revised"] for p in peer_rows) if injected else None,
        revision_and_final_target=bool(any(p["revised"] for p in peer_rows) and selected) if injected else None,
        peer_count=len(peer_rows), exposed_peers=sum(p["exposed"] for p in peer_rows), revised_peers=sum(p["revised"] for p in peer_rows),
        baseline_target_peers=sum(p["baseline_target"] for p in peer_rows), no_direct_revision_peers=sum(p["no_direct_revision"] for p in peer_rows),
        correct_to_target_peers=sum(p["correct_to_target"] for p in peer_rows), revised_final_target_peers=sum(p["revised"] and p["final_target"] for p in peer_rows),
        eligible_voters=len(ballots), blocked_final_voters=len(blocked), eligible_peer_target_votes=len(support),
        source_vote_eligible=source in eligible_ids if injected else "", winning_count=count, highest_count_tied=tied, strict_majority=majority,
        path_category=category, replay_choice=choice, replay_status=status, replay_ok=True,
        remove_source_choice=removed if injected else "", remove_source_loses_target=bool(selected and not same(removed, target)) if injected else "",
        legacy_attrret=raw["propagation_metrics"].get("final_error_retention_causal"))
    for field, value in result.items():
        check_equal("" if value is None else value, row[field], f"{row['trace']}:{row['source_line']} {field}")
    result.update(family=row["family"], condition=condition, case_id=row["case_id"], trace=row["trace"])
    return result, peer_rows, ballots, raw, ref


def audit():
    cases = read_csv(EVIDENCE / "case_level_event_table.csv")
    completion = read_csv(EVIDENCE / "reanalysis_completion.csv")
    arms = read_csv(EVIDENCE / "reanalysis_arm_results.csv")
    pairs = read_csv(EVIDENCE / "reanalysis_paired_results.csv")
    hashes = {}
    for name in ("source_snapshot_sha256.csv", "reanalysis_input_sha256.csv"):
        for entry in read_csv(EVIDENCE / name):
            path = ROOT / entry["path"]
            check_equal(sha(path), entry["sha256"], f"input integrity {entry['path']}")
            hashes[entry["path"]] = entry["sha256"]
    traces = {}
    pools = {}
    latest = {}
    for trace in {r["trace"] for r in cases}:
        with (ROOT / trace).open(encoding="utf-8-sig") as stream:
            records = {line: json.loads(text) for line, text in enumerate(stream, 1) if text.strip()}
        traces[trace] = records
        latest[trace] = {r["case_id"]: (line, r) for line, r in records.items()}
    for path in {r["reference_csv"] for r in cases}:
        pools[path] = {r.get("pilot_case_id") or r.get("case_uid") or r.get("source_record_id"): r for r in read_csv(ROOT / path)}
    for row in completion:
        item = latest[row["trace"]].get(row["case_id"])
        status = "not_run" if item is None else ("complete" if item[1].get("run_status") != "failed" and isinstance(item[1].get("propagation_metrics"), dict) else "failed")
        check_equal(status, row["status"], "last-record selection")
        check_equal(item[0] if item else "", row["source_line"], "selected line")
    peer_index = {(p["trace"], p["case_id"], p["agent_id"]): p for p in read_csv(EVIDENCE / "peer_level_event_audit.csv")}
    rebuilt, details = [], {}
    peers_seen = 0
    for row in cases:
        raw = traces[row["trace"]][int(row["source_line"])]
        check_equal(raw["case_id"], row["case_id"], "source identifier")
        if not raw.get("injection"):
            # Input rows are not assumed to be sorted by condition.
            contaminated_row = next((r for r in cases if r["family"] == row["family"] and r["condition"] == "debate_injected" and r["case_id"] == row["case_id"]), None)
            target = ""
            if contaminated_row:
                contaminated_raw = traces[contaminated_row["trace"]][int(contaminated_row["source_line"])]
                target = contaminated_raw["injection"]["wrong_diagnosis"]
            check_equal(target, row["target"], "clean borrowed target linkage")
        result, peers, ballots, raw, ref = reconstruct(row, raw, pools[row["reference_csv"]][row["case_id"]])
        for peer in peers:
            stored = peer_index[row["trace"], row["case_id"], peer["agent_id"]]
            for field, value in peer.items():
                check_equal(value, stored[field], f"peer {row['case_id']} {field}")
        peers_seen += len(peers)
        rebuilt.append(result)
        details[row["family"], row["condition"], row["case_id"]] = (row, raw, ref, ballots)
    check_equal(peers_seen, len(peer_index), "peer audit coverage")
    groups = defaultdict(list)
    for row in rebuilt:
        groups[row["family"], row["condition"]].append(row)
    for arm in arms:
        selected = subset_rows(groups[arm["family"], arm["condition"]], arm["subset"])
        comp = [r for r in completion if r["trace"] == arm["trace"]]
        for field, value in dict(planned_n=len(comp), complete_n=sum(r["status"] == "complete" for r in comp),
                failed_n=sum(r["status"] == "failed" for r in comp), not_run_n=sum(r["status"] == "not_run" for r in comp),
                subset_n=len(selected), gate_flags_inferred=sum(r["gate_flags_origin"] == "condition_inference" for r in selected)).items():
            check_equal(value, arm[field], f"arm {field}")
        for endpoint in ENDPOINTS + ("abstain", "any_exposure"):
            values = [r[endpoint] for r in selected if r[endpoint] is not None]
            check_equal(len(values), arm[endpoint+"_n"], "arm denominator")
            check_equal(sum(values) if values else "", arm[endpoint+"_events"], "arm events")
            if values:
                n, k = len(values), sum(values)
                z = 1.959963984540054
                rate, den = k/n, 1+z*z/n
                center = (rate+z*z/(2*n))/den
                radius = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
                near(arm[endpoint+"_rate"], rate, "arm rate")
                near(arm[endpoint+"_ci_low"], max(0, center-radius), "Wilson lower")
                near(arm[endpoint+"_ci_high"], min(1, center+radius), "Wilson upper")
        for category in CATEGORIES:
            check_equal(sum(r["path_category"] == category for r in selected), arm[category], "arm partition")
        check_equal(sum(r["path_category"] == "source_only" and r["highest_count_tied"] for r in selected), arm["source_only_tied"], "source ties")
        check_equal(sum(r["remove_source_loses_target"] is True for r in selected), arm["source_removal_loses_target"], "source removal")
        check_equal(sum(r["legacy_attrret"] is True for r in selected), arm["legacy_attrret_events"], "historical metric")
    raw_ps = []
    for pair in pairs:
        a = {r["case_id"]: r for r in groups[pair["family"], pair["condition_a"]]}
        b = {r["case_id"]: r for r in groups[pair["family"], pair["condition_b"]]}
        pa = {r["case_id"] for r in completion if (r["family"], r["condition"]) == (pair["family"], pair["condition_a"])}
        pb = {r["case_id"] for r in completion if (r["family"], r["condition"]) == (pair["family"], pair["condition_b"])}
        planned = pa & pb
        if pair["subset"] != "all_retained":
            eligible = {cid for cid in planned if a.get(cid, {}).get("wrong_option_eligible") is True and b.get(cid, {}).get("wrong_option_eligible") is True}
            planned = eligible if pair["subset"] == "wrong_option_eligible" else planned-eligible
        endpoint = pair["endpoint"]
        ids = sorted(cid for cid in planned & a.keys() & b.keys() if a[cid][endpoint] is not None and b[cid][endpoint] is not None and (endpoint == "final_correct" or same(a[cid]["target"], b[cid]["target"])))
        check_equal(";".join(ids), pair["complete_case_ids"], "complete pair IDs")
        counts = Counter((a[cid][endpoint], b[cid][endpoint]) for cid in ids)
        n, x, y = len(ids), counts[False, True], counts[True, False]
        for field, value in dict(n_pairs=n, n00=counts[False, False], n01=x, n10=y, n11=counts[True, True],
                events_a=y+counts[True, True], events_b=x+counts[True, True], planned_n=len(planned)).items():
            check_equal(value, pair[field], f"paired {field}")
        near(pair["rd_b_minus_a"], (x-y)/n, "paired difference")
        lx, ux = cp(x, n)
        ly, uy = cp(y, n)
        near(pair["rd_ci_low"], lx-uy, "paired interval lower")
        near(pair["rd_ci_high"], ux-ly, "paired interval upper")
        discordant = x+y
        p = min(1, math.fsum(math.comb(discordant, j) / 2**discordant for j in range(min(x, y)+1))*2) if discordant else 1
        near(pair["mcnemar_exact_p"], p, "McNemar")
        raw_ps.append(p)
        va = [a[cid][endpoint] for cid in planned & a.keys() if a[cid][endpoint] is not None]
        vb = [b[cid][endpoint] for cid in planned & b.keys() if b[cid][endpoint] is not None]
        check_equal(len(va), pair["observed_a_n"], "observed A")
        check_equal(len(vb), pair["observed_b_n"], "observed B")
        near(pair["missing_outcome_rd_low"], (sum(vb)-sum(va)-(len(planned)-len(va)))/len(planned), "missing lower")
        near(pair["missing_outcome_rd_high"], (sum(vb)+len(planned)-len(vb)-sum(va))/len(planned), "missing upper")
    order = sorted(range(len(pairs)), key=lambda i: raw_ps[i])
    maximum = 0.0
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1, raw_ps[i]*(len(pairs)-rank)))
        near(pairs[i]["holm_global_p"], maximum, "global Holm")
        check_equal(len(pairs), pairs[i]["holm_family_size"], "Holm family size")
    research = [r for r in rebuilt if r["family"] != "FORMAT_FAILOVER_DRILL"]
    check_equal(len(research), 1548, "manuscript research records")
    check_equal(len({r["trace"] for r in research}), 25, "manuscript research traces")
    check_equal(Counter(r["status"] for r in completion if r["family"] != "FORMAT_FAILOVER_DRILL"), Counter(complete=1548, failed=10, not_run=26), "research completion")
    check_equal(sum(r["injected"] for r in research), 1199, "research contaminated records")
    check_equal(sum(r["gate_flags_origin"] == "condition_inference" for r in research), 240, "inferred switches")
    return dict(cases=cases, research=research, arms=arms, pairs=pairs, details=details, hashes=hashes, peers=peers_seen)



def main():
    data = audit()
    print(json.dumps(dict(status="passed", research_records=len(data["research"]),
        raw_records_checked=len(data["cases"]), peer_records_checked=data["peers"],
        arm_rows_checked=len(data["arms"]), paired_rows_checked=len(data["pairs"]),
        unique_input_hashes_verified=len(data["hashes"]), new_model_calls=0)))


if __name__ == "__main__":
    main()
