"""Retrospective trace reanalysis, standard library only; no model calls.

Rules: paper/rewrite/回顾性重分析定义与处理协议_v1.md.
Historical traces are read-only. All generated outputs go to rewrite/evidence.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from build_rewrite_evidence_map import (
    ROOT, OUT, TRACE_DIR, DATA_DIR, canonical_file, canonical_gold,
    digest, family, key, read_jsonl, rel, same, successful, write_csv,
)

PROTOCOL = ROOT / "protocol/retrospective_reanalysis_protocol_original.md"
ENDPOINTS = ("final_target", "final_correct", "any_revision", "revision_and_final_target")


def read_pool(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {r.get("pilot_case_id") or r.get("case_uid") or r.get("source_record_id"): r
                for r in csv.DictReader(stream)}


def gate_flags(r):
    gated = bool(r.get("evidence_gate"))
    condition = r["condition_id"]
    inferred = "gate_peer_visibility" not in r or "gate_final_vote" not in r
    return (
        bool(r.get("gate_peer_visibility", gated and not condition.endswith("_vote_only"))),
        bool(r.get("gate_final_vote", gated and not condition.endswith("_visibility_only"))),
        "condition_inference" if inferred else "recorded",
    )


def ballot_result(turns):
    """Exact-label plurality, first occurrence breaks ties; no semantic clustering."""
    if not turns:
        return "", 0, "no_ballots", False, False
    counts = Counter(key(t["primary_diagnosis"]) for t in turns)
    winner, count = counts.most_common(1)[0]
    label = next(t["primary_diagnosis"] for t in turns if key(t["primary_diagnosis"]) == winner)
    majority = count > len(turns) / 2
    tie = sum(c == count for c in counts.values()) > 1
    return label, count, "majority_vote" if majority else "plurality_tiebreak", tie, majority


def analyze_record(path, line, r, ref, matched_target=None):
    """Return one case row, peer rows and audit issue strings."""
    gold, options = canonical_gold(ref)
    issues = []
    turns = r.get("turns") or []
    by_agent = defaultdict(list)
    for t in turns:
        by_agent[t["agent_id"]].append(t)
    for ts in by_agent.values():
        ts.sort(key=lambda t: t["round_index"])
    rounds = r["rounds"]
    if not by_agent or any([t["round_index"] for t in ts] != list(range(1, rounds + 1))
                           for ts in by_agent.values()):
        raise ValueError(f"incomplete/duplicate rounds: {rel(path)}:{line}")
    injection = r.get("injection")
    source = injection["source_agent_id"] if injection else ""
    target = injection["wrong_diagnosis"] if injection else (matched_target or "")
    target_origin = "actual_injection" if injection else ("matched_injected_arm" if target else "not_applicable")
    injection_valid = bool(injection and any(same(target, o) for o in options) and not same(target, gold))
    if injection:
        if source not in by_agent:
            raise ValueError("source missing from turns")
        for t in by_agent[source]:
            applies = (t["round_index"] >= injection["round_index"] if injection.get("persist")
                       else t["round_index"] == injection["round_index"])
            if applies != bool(t.get("injected")) or (applies and not same(t["primary_diagnosis"], target)):
                issues.append("injection_schedule_mismatch")
    vis, vote, flags_origin = gate_flags(r)
    gates = r.get("gate_assessments") or []
    if r.get("evidence_gate"):
        expected_gate_keys = {(t["round_index"], t["agent_id"]) for t in turns}
        actual_gate_keys = [(g["round_index"], g["target_agent_id"]) for g in gates]
        if set(actual_gate_keys) != expected_gate_keys or len(actual_gate_keys) != len(expected_gate_keys):
            issues.append("missing_duplicate_or_extra_gate_assessments")
    for t in turns:
        if t["round_index"] == 1 and t.get("visible_peer_ids"):
            issues.append("unexpected_round1_visibility")
        if t["round_index"] > 1:
            blocked = {g["target_agent_id"] for g in gates
                       if g["round_index"] == t["round_index"] - 1 and g["blocking"]} if vis else set()
            expected = set(by_agent) - {t["agent_id"]} - blocked
            if r["topology"] == "complete" and set(t.get("visible_peer_ids", [])) != expected:
                issues.append("visibility_reconstruction_mismatch")

    # First occurrence preserves the actual final-turn vote order.
    final_turns = [t for t in turns if t["round_index"] == rounds]
    blocked = {g["target_agent_id"] for g in gates if g["round_index"] == rounds and g["blocking"]} if vote else set()
    eligible = [t for t in final_turns if t["agent_id"] not in blocked]
    if r["decision_protocol"] == "first_agent":
        eligible = final_turns[:1]
        replay, count, status, tie, majority = eligible[0]["primary_diagnosis"], 1, "first_agent", False, True
    else:
        replay, count, status, tie, majority = ballot_result(eligible)
        if not eligible:
            status = "abstain_all_blocked"
        elif blocked:
            status += "_gate_filtered"
    replay_ok = (key(replay) == key(r.get("final_diagnosis"))
                 and count == r.get("final_vote_count") and status == r.get("final_status"))
    if not replay_ok:
        issues.append("final_vote_replay_mismatch")
    final = r.get("final_diagnosis") or ""
    correct = same(final, gold)
    if bool(r["propagation_metrics"].get("final_correct")) != correct:
        issues.append("stored_correctness_mismatch")
    eligible_ids = {t["agent_id"] for t in eligible}
    peers = []
    if injection:
        for aid, ts in by_agent.items():
            if aid == source:
                continue
            exposed, revised, ungated_revised, correct_revised = [], [], [], []
            for index, t in enumerate(ts):
                ri = t["round_index"]
                src_prior = next((s for s in by_agent[source] if s["round_index"] == ri - 1), None)
                direct = (source in t.get("visible_peer_ids", []) and src_prior is not None
                          and bool(src_prior.get("injected")) and same(src_prior["primary_diagnosis"], target))
                if direct:
                    exposed.append(ri)
                prior = ts[index - 1] if index else None
                changed = prior is not None and not same(prior["primary_diagnosis"], target) and same(t["primary_diagnosis"], target)
                if changed:
                    (revised if direct else ungated_revised).append(ri)
                    if direct and same(prior["primary_diagnosis"], gold):
                        correct_revised.append(ri)
            peers.append(dict(
                trace=rel(path), source_line=line, family=family(path.name), case_id=r["case_id"],
                agent_id=aid, source_agent=source, target=target,
                baseline_choice=ts[0]["primary_diagnosis"], baseline_no_peers=not ts[0].get("visible_peer_ids"),
                baseline_target=same(ts[0]["primary_diagnosis"], target),
                direct_exposure_rounds=";".join(map(str, exposed)),
                post_exposure_revision_rounds=";".join(map(str, revised)),
                no_direct_exposure_revision_rounds=";".join(map(str, ungated_revised)),
                correct_to_target_revision_rounds=";".join(map(str, correct_revised)),
                exposed=bool(exposed), revised=bool(revised), no_direct_revision=bool(ungated_revised),
                correct_to_target=bool(correct_revised),
                final_choice=ts[-1]["primary_diagnosis"], final_target=same(ts[-1]["primary_diagnosis"], target),
                eligible_final_vote=aid in eligible_ids,
            ))
    final_target = same(final, target) if target else None
    n_exposed = sum(p["exposed"] for p in peers)
    n_revised = sum(p["revised"] for p in peers)
    support = [p for p in peers if p["eligible_final_vote"] and p["final_target"]]
    path_category = "not_target_selected" if target else "not_applicable"
    if injection and final_target:
        if not replay_ok:
            path_category = "unresolved_replay_conflict"
        elif not support:
            path_category = "source_only"
        elif any(p["revised"] for p in support):
            path_category = "peer_post_exposure_revision_supported"
        elif any(p["baseline_target"] for p in support):
            path_category = "peer_baseline_same_option_supported"
        else:
            path_category = "peer_other_same_option_supported"
    elif not injection and final_target:
        path_category = "matched_target_control"
    without_source = [t for t in eligible if t["agent_id"] != source]
    removed, _, _, _, _ = ballot_result(without_source)
    pm = r["propagation_metrics"]
    result = dict(
        trace=rel(path), source_line=line, family=family(path.name), condition=r["condition_id"],
        case_id=r["case_id"], model=";".join(sorted(set(r["model_ids"]))), reference_csv=rel(canonical_file(path.name)),
        reference=gold, stored_gold=r.get("gold_diagnosis", ""), target=target, target_origin=target_origin,
        injected=bool(injection), wrong_option_eligible=injection_valid if injection else None,
        target_equals_reference=same(target, gold) if target else None, source_agent=source,
        rounds=rounds, observed_turns=len(turns), final_choice=final, final_correct=correct,
        final_target=final_target, abstain=not bool(final.strip()),
        final_outside_options=bool(final) and not any(same(final, o) for o in options),
        gate_backend=r.get("gate_backend"), gate_visibility=vis, gate_vote=vote, gate_flags_origin=flags_origin,
        verbatim_prompt_from_current_code=bool(r.get("evidence_gate") and str(r.get("gate_backend")).startswith("hard")),
        peer_count=len(peers), exposed_peers=n_exposed, baseline_target_peers=sum(p["baseline_target"] for p in peers),
        revised_peers=n_revised, no_direct_revision_peers=sum(p["no_direct_revision"] for p in peers),
        correct_to_target_peers=sum(p["correct_to_target"] for p in peers),
        revised_final_target_peers=sum(p["revised"] and p["final_target"] for p in peers),
        any_exposure=bool(n_exposed) if injection else None, any_revision=bool(n_revised) if injection else None,
        revision_and_final_target=bool(n_revised and final_target) if injection else None,
        eligible_voters=len(eligible), blocked_final_voters=len(blocked),
        eligible_peer_target_votes=len(support), source_vote_eligible=source in eligible_ids if injection else None,
        winning_count=count, highest_count_tied=tie, strict_majority=majority, path_category=path_category,
        eligible_ballots_json=json.dumps([{t["agent_id"]: t["primary_diagnosis"]} for t in eligible], ensure_ascii=False),
        replay_choice=replay, replay_status=status, replay_ok=replay_ok,
        remove_source_choice=removed if injection else "", remove_source_no_votes=not without_source if injection else None,
        remove_source_changes_answer=(key(removed) != key(final)) if injection else None,
        remove_source_loses_target=bool(final_target and not same(removed, target)) if injection else None,
        legacy_attrret=pm.get("final_error_retention_causal"),
        legacy_peer_flip=pm.get("causal_primary_adopting_agents"),
        legacy_exposed=pm.get("exposed_agents"),
    )
    return result, peers, sorted(set(issues))


def wilson(k, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, mid - radius), min(1, mid + radius)


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            + j * math.log(p) + (n - j) * math.log1p(-p) for j in range(k + 1)]
    peak = max(logs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(x - peak) for x in logs))


def invert_cdf(k, n, value):
    low, high = 0.0, 1.0
    for _ in range(70):
        mid = (low + high) / 2
        if binomial_cdf(k, n, mid) > value:
            low = mid
        else:
            high = mid
    return (low + high) / 2


@lru_cache(maxsize=None)
def clopper_pearson(k, n, alpha=0.025):
    return (0.0 if k == 0 else invert_cdf(k - 1, n, 1 - alpha / 2),
            1.0 if k == n else invert_cdf(k, n, alpha / 2))


def paired_stats(pairs):
    n = len(pairs)
    counts = Counter((bool(a), bool(b)) for a, b in pairs)
    n01, n10 = counts[False, True], counts[True, False]
    if not n:
        raise ValueError("no complete pairs")
    l01, u01 = clopper_pearson(n01, n)
    l10, u10 = clopper_pearson(n10, n)
    m = n01 + n10
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(min(n01, n10) + 1)) / (2 ** m)) if m else 1.0
    return dict(n_pairs=n, n00=counts[False, False], n01=n01, n10=n10, n11=counts[True, True],
                events_a=n10 + counts[True, True], events_b=n01 + counts[True, True],
                rd_b_minus_a=(n01 - n10) / n, rd_ci_low=l01 - u10, rd_ci_high=u01 - l10,
                ci_method="CP97.5_discordant_Bonferroni_conservative95", mcnemar_exact_p=p)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def planned_ids(group, condition, pools, arms):
    if group.startswith("MED_EXT59"):
        name = "medxpertqa_text_test60_external.csv" if condition == "single" and group == "MED_EXT59_GLM" else "medxpertqa_text_test60_external_n59.csv"
        return set(pools[name])
    if group.startswith("MED_EXT91"):
        return set(pools["medxpertqa_text_ext150_expansion91.csv"])
    if group.startswith("ARENA60"):
        return {s.strip() for s in (DATA_DIR / "propagation_holdout60_case_ids.txt").read_text(encoding="utf-8-sig").splitlines() if s.strip()}
    return set(arms)


def subsets(rows):
    yield "all_retained", rows
    if any(r["injected"] for r in rows):
        yield "wrong_option_eligible", [r for r in rows if r["wrong_option_eligible"]]
        invalid = [r for r in rows if r["injected"] and not r["wrong_option_eligible"]]
        if invalid:
            yield "injection_design_violation", invalid


def comparison_specs(group):
    inj, hard = "debate_injected", "debate_injected_hardgate"
    if group.startswith("MED_EXT59"):
        return [(inj, hard), ("debate_clean", "debate_clean_hardgate"), ("debate_clean", inj),
                ("debate_clean_hardgate", hard), ("single", "debate_clean")]
    if group == "ARENA60_DS_FLASH":
        return [(inj, hard), (inj, hard + "_visibility_only"), (inj, hard + "_vote_only"),
                (hard + "_visibility_only", hard), (hard + "_vote_only", hard),
                (hard + "_marker", hard), (hard + "_extractive", hard)]
    if group.startswith("ARENA60") or group == "MED_EXT91_DS":
        return [(inj, hard)]
    return []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze verification occurs BEFORE endpoint generation, and again after.
    with (OUT / "source_snapshot_sha256.csv").open(encoding="utf-8-sig", newline="") as stream:
        prior_hashes = list(csv.DictReader(stream))
    if any(not (ROOT / h["path"]).is_file() or digest(ROOT / h["path"]) != h["sha256"] for h in prior_hashes):
        raise RuntimeError("First-stage input snapshot changed; review before reanalysis.")
    pools = {p.name: read_pool(p) for p in DATA_DIR.glob("*.csv")}
    raw_arms, all_cases, all_peers, issues, completion = {}, [], [], [], []
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        records, errors, nonobjects = read_jsonl(path)
        if errors or nonobjects:
            raise ValueError(f"JSONL schema/parse failure in {path}")
        latest = {r["case_id"]: (line, r) for line, r in records}
        raw_arms[path] = latest
    matched = {}
    for path, arm in raw_arms.items():
        for cid, (_, r) in arm.items():
            if successful(r) and r["condition_id"] == "debate_injected" and r.get("injection"):
                matched[family(path.name), cid] = r["injection"]["wrong_diagnosis"]
    kept = defaultdict(dict)
    arm_rows = []
    input_paths = {PROTOCOL, Path(__file__).resolve(), ROOT / "analysis/build_rewrite_evidence_map.py",
                   DATA_DIR / "propagation_holdout60_case_ids.txt", DATA_DIR / "medxpertqa_text_test60_external_n59.csv"}
    for path, arm in raw_arms.items():
        group = family(path.name)
        condition = next(iter(arm.values()))[1]["condition_id"]
        ids = planned_ids(group, condition, pools, arm)
        if set(arm) - ids:
            raise ValueError(f"Unexpected IDs in {path}")
        refpath = canonical_file(path.name)
        input_paths.update({path, refpath})
        rows = []
        for cid in sorted(ids):
            item = arm.get(cid)
            ok = item is not None and successful(item[1])
            completion.append(dict(trace=rel(path), family=group, condition=condition, case_id=cid,
                                   source_line=item[0] if item else None,
                                   status="complete" if ok else ("failed" if item else "not_run"),
                                   failure_type=item[1].get("error_type", "missing_metrics") if item and not ok else ""))
            if not ok:
                continue
            line, record = item
            case, peers, case_issues = analyze_record(path, line, record, pools[refpath.name][cid], matched.get((group, cid)))
            rows.append(case)
            all_cases.append(case)
            all_peers.extend(peers)
            kept[group, condition][cid] = case
            for issue in case_issues:
                issues.append(dict(trace=rel(path), source_line=line, case_id=cid, issue=issue))
        for subset, selected in subsets(rows):
            summary = dict(trace=rel(path), family=group, condition=condition, subset=subset,
                           planned_n=len(ids), attempted_n=len(arm), complete_n=len(rows),
                           failed_n=sum(not successful(r) for _, r in arm.values()), not_run_n=len(ids - set(arm)),
                           subset_n=len(selected), gate_flags_inferred=sum(r["gate_flags_origin"] == "condition_inference" for r in selected))
            for endpoint in ENDPOINTS + ("abstain", "any_exposure"):
                values = [r[endpoint] for r in selected if r[endpoint] is not None]
                k = sum(values)
                lo, hi = wilson(k, len(values))
                summary.update({endpoint + "_n": len(values), endpoint + "_events": k if values else None,
                                endpoint + "_rate": k / len(values) if values else None,
                                endpoint + "_ci_low": lo, endpoint + "_ci_high": hi})
            for category in ("source_only", "peer_post_exposure_revision_supported", "peer_baseline_same_option_supported", "peer_other_same_option_supported"):
                summary[category] = sum(r["path_category"] == category for r in selected)
            summary.update(legacy_attrret_events=sum(r["legacy_attrret"] is True for r in selected),
                           source_only_tied=sum(r["path_category"] == "source_only" and r["highest_count_tied"] for r in selected),
                           source_removal_loses_target=sum(r["remove_source_loses_target"] is True for r in selected),
                           revised_peers=sum(r["revised_peers"] for r in selected),
                           exposed_peers=sum(r["exposed_peers"] for r in selected),
                           baseline_target_peers=sum(r["baseline_target_peers"] for r in selected))
            arm_rows.append(summary)

    paired, coverage = [], []
    for group in sorted({g for g, _ in kept}):
        for ca, cb in comparison_specs(group):
            a, b = kept[group, ca], kept[group, cb]
            # Cross-arm single comparisons use the common planned 59, not single 60.
            pa = planned_ids(group, ca, pools, a)
            pb = planned_ids(group, cb, pools, b)
            plan = pa & pb
            common = plan & set(a) & set(b)
            scope = dict(family=group, condition_a=ca, condition_b=cb)
            coverage.append(dict(**scope, planned_n=len(plan), common_complete_n=len(common),
                                 a_only_complete=";".join(sorted((plan & set(a)) - set(b))),
                                 b_only_complete=";".join(sorted((plan & set(b)) - set(a))),
                                 neither_complete=";".join(sorted(plan - set(a) - set(b)))))
            stratum_sets = [("all_retained", plan)]
            if group == "MED_EXT91_DS":
                eligible_ids = {cid for cid in plan if a.get(cid, {}).get("wrong_option_eligible") is True
                                and b.get(cid, {}).get("wrong_option_eligible") is True}
                stratum_sets += [("wrong_option_eligible", eligible_ids), ("injection_design_violation", plan - eligible_ids)]
            for subset, subset_plan in stratum_sets:
                for endpoint in ENDPOINTS:
                    complete = [cid for cid in sorted(common & subset_plan)
                                if a[cid][endpoint] is not None and b[cid][endpoint] is not None
                                and (endpoint == "final_correct" or same(a[cid]["target"], b[cid]["target"]))]
                    if not complete:
                        continue
                    stats = paired_stats([(a[cid][endpoint], b[cid][endpoint]) for cid in complete])
                    values_a = [a[cid][endpoint] for cid in subset_plan & set(a) if a[cid][endpoint] is not None]
                    values_b = [b[cid][endpoint] for cid in subset_plan & set(b) if b[cid][endpoint] is not None]
                    na, nb, np = len(values_a), len(values_b), len(subset_plan)
                    paired.append(dict(**scope, subset=subset, endpoint=endpoint, **stats, planned_n=np,
                                       observed_a_n=na, observed_b_n=nb,
                                       missing_outcome_rd_low=(sum(values_b) - sum(values_a) - (np - na)) / np,
                                       missing_outcome_rd_high=(sum(values_b) + (np - nb) - sum(values_a)) / np,
                                       complete_case_ids=";".join(complete)))
    adjusted = holm([p["mcnemar_exact_p"] for p in paired])
    for p, adj in zip(paired, adjusted):
        p["holm_global_p"] = adj
        p["holm_family_size"] = len(paired)
        p["analysis_status"] = "retrospective_exploratory"

    write_csv("case_level_event_table.csv", all_cases)
    write_csv("peer_level_event_audit.csv", all_peers)
    write_csv("reanalysis_completion.csv", completion)
    write_csv("reanalysis_arm_results.csv", arm_rows)
    write_csv("reanalysis_paired_results.csv", paired)
    write_csv("reanalysis_pair_coverage.csv", coverage)
    # Always emit an empty audit with a schema, so absence is not ambiguous.
    with (OUT / "reanalysis_issues.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trace", "source_line", "case_id", "issue"])
        writer.writeheader()
        writer.writerows(issues)
    invalid = [r for r in all_cases if r["injected"] and not r["wrong_option_eligible"]]
    write_csv("reanalysis_injection_violations.csv", invalid)
    for h in prior_hashes:
        if digest(ROOT / h["path"]) != h["sha256"]:
            raise RuntimeError("Historical input modified during analysis")
    write_csv("reanalysis_input_sha256.csv", [dict(path=rel(p), sha256=digest(p)) for p in sorted(input_paths)])
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=rel(PROTOCOL), protocol_sha256=digest(PROTOCOL),
                   successful_case_arm_records=len(all_cases), peer_arm_records=len(all_peers),
                   completion_rows=len(completion), issues=len(issues), issue_types=dict(Counter(r["issue"] for r in issues)),
                   invalid_injection_records=len(invalid), paired_tests=len(paired),
                   historical_input_hashes_verified=len(prior_hashes), historical_inputs_unchanged=True,
                   new_model_calls=0, statistics="Wilson95; conservative paired CP/Bonferroni95; exact McNemar; global Holm",
                   raw_input_scope="local_inputs/outputs/propagation/*.jsonl")
    (OUT / "reanalysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if issues:
        raise RuntimeError("Audit conflicts require review before manuscript use")


if __name__ == "__main__":
    main()
