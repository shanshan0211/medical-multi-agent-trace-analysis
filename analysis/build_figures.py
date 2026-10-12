"""Export the paper results figure from regenerated evidence; no model calls.

This release never edits the manuscript source.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper" / "rewrite" / "evidence"


def rows(name):
    with (EVIDENCE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select(records, **criteria):
    matches = [r for r in records if all(r[k] == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}; found {len(matches)}")
    return matches[0]


def figure_results():
    arms = rows("reanalysis_arm_results.csv")
    paired = rows("reanalysis_paired_results.csv")
    cases = rows("case_level_event_table.csv")
    paths = (
        ("source_only", "sourcecolor", "Source-only tie"),
        ("peer_post_exposure_revision_supported", "revisioncolor", "Revision-supported"),
        ("peer_baseline_same_option_supported", "baselinecolor", "Baseline agreement"),
    )
    families = (("MED_EXT59_DS", "DeepSeek"), ("MED_EXT59_GLM", "GLM"))
    data = []
    out = [
        r"% Generated from audited CSVs; run scripts/build_figures.py to refresh.",
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\small,text=inkcolor]",
        r"\node[anchor=west,font=\bfseries] at (0,0) {A\quad Different ballot pathways lead to the same target selection};",
        r"\node[anchor=west,font=\footnotesize] at (0,-0.45) {Bars partition selected targets; row labels retain the completed-case denominator.};",
    ]
    for index, (family, label) in enumerate(families):
        row = select(arms, family=family, condition="debate_injected", subset="all_retained")
        n, events = int(row["subset_n"]), int(row["final_target_events"])
        selected = [r for r in cases if r["family"] == family and r["condition"] == "debate_injected"]
        assert len(selected) == n
        assert sum(r["final_target"] == "True" for r in selected) == events
        assert int(row["source_only"]) == int(row["source_only_tied"])
        assert int(row["peer_other_same_option_supported"]) == 0
        y, start = -1.15 - index * 0.95, 4.05
        out.append(rf"\node[anchor=east] at (3.85,{y:.3f}) {{{label}: {events}/{n}}};")
        for category, color, _ in paths:
            count = int(row[category])
            assert sum(r["path_category"] == category for r in selected) == count
            end = start + 0.55 * count
            out.extend([
                rf"\fill[{color}] ({start:.3f},{y - 0.24:.3f}) rectangle ({end:.3f},{y + 0.24:.3f});",
                rf"\node[text=white,font=\small\bfseries] at ({(start + end) / 2:.3f},{y:.3f}) {{{count}}};",
            ])
            start = end
            data.append(dict(panel="A", family=family, condition_a="debate_injected", condition_b="",
                             endpoint=category, n=n, events_a=count, events_b="", rd_pp="",
                             ci_low_pp="", ci_high_pp="", holm_p=""))
        assert sum(int(row[p[0]]) for p in paths) == events
    out.append(r"\draw[inkcolor!60] (4.05,-2.60) -- (13.95,-2.60);")
    for tick in range(0, 19, 3):
        x = 4.05 + 0.55 * tick
        out.append(rf"\draw[inkcolor!60] ({x:.3f},-2.60) -- ({x:.3f},-2.68) node[below,font=\footnotesize] {{{tick}}};")
    out.append(r"\node[font=\footnotesize] at (9,-3.18) {Number of target-selected cases};")
    for index, (_, color, label) in enumerate(paths):
        x = index * 5.0
        out.append(rf"\fill[{color}] ({x:.2f},-3.73) rectangle ({x + .28:.2f},-3.45);")
        out.append(rf"\node[anchor=west,font=\footnotesize] at ({x + .38:.2f},-3.59) {{{label}}};")
    out.extend([
        r"\node[anchor=west,font=\bfseries] at (0,-4.50) {B\quad Paired target selection and correctness have different uncertainty};",
        r"\node[anchor=east,font=\footnotesize] at (14.6,-5.10) {Holm $p$};",
        r"\draw[dashed,inkcolor!60] (8.6,-5.45) -- (8.6,-10.15);",
    ])
    settings = [
        ("debate_injected", "final_target", "target", "sourcecolor"),
        ("debate_injected", "final_correct", "correct, contaminated", "revisioncolor"),
        ("debate_clean", "final_correct", "correct, clean", "revisioncolor"),
    ]
    for group, (condition, endpoint, title, color) in enumerate(settings):
        for model_index, (family, label) in enumerate(families):
            row = select(paired, family=family, condition_a=condition,
                         condition_b=condition + "_hardgate", subset="all_retained", endpoint=endpoint)
            assert row["holm_family_size"] == "76"
            assert row["ci_method"] == "CP97.5_discordant_Bonferroni_conservative95"
            n, a, b = (int(row[k]) for k in ("n_pairs", "events_a", "events_b"))
            estimate, lo, hi = (100 * float(row[k]) for k in ("rd_b_minus_a", "rd_ci_low", "rd_ci_high"))
            assert abs(estimate - 100 * (b - a) / n) < 1e-9
            y = -5.75 - group * 1.62 - model_index * 0.66
            xp, xl, xh = (8.6 + value * 0.125 for value in (estimate, lo, hi))
            out.extend([
                rf"\node[anchor=east,font=\footnotesize,align=right] at (3.95,{y:.3f}) {{{label}, {title}\\$n={n}$}};",
                rf"\draw[{color},line width=0.9pt] ({xl:.4f},{y:.3f}) -- ({xh:.4f},{y:.3f});",
                rf"\draw[{color}] ({xl:.4f},{y-.08:.3f}) -- ({xl:.4f},{y+.08:.3f});",
                rf"\draw[{color}] ({xh:.4f},{y-.08:.3f}) -- ({xh:.4f},{y+.08:.3f});",
                rf"\fill[{color}] ({xp:.4f},{y:.3f}) circle (2pt);",
                rf"\node[anchor=east,font=\footnotesize] at (14.6,{y:.3f}) {{{float(row['holm_global_p']):.3f}}};",
            ])
            data.append(dict(panel="B", family=family, condition_a=condition, condition_b=condition + "_hardgate",
                             endpoint=endpoint, n=n, events_a=a, events_b=b, rd_pp=estimate,
                             ci_low_pp=lo, ci_high_pp=hi, holm_p=row["holm_global_p"]))
    out.append(r"\draw[inkcolor!60] (4.225,-10.30) -- (11.725,-10.30);")
    for tick in (-30, -20, -10, 0, 10, 20):
        x = 8.6 + tick * .125
        out.append(rf"\draw[inkcolor!60] ({x:.3f},-10.30) -- ({x:.3f},-10.38) node[below,font=\footnotesize] {{$ {tick} $}};")
    out.extend([
        r"\node[font=\footnotesize] at (8.0,-10.98) {Full gate minus ungated (percentage points)};",
        r"\end{tikzpicture}",
    ])
    return "\n".join(out), data


def main():
    figure, data = figure_results()
    target = ROOT / "generated_figures"
    target.mkdir(exist_ok=True)
    preamble = (
        r"\documentclass[tikz,border=5pt]{standalone}" + "\n"
        + r"\usepackage[T1]{fontenc}" + "\n"
        + r"\definecolor{sourcecolor}{HTML}{B45332}" + "\n"
        + r"\definecolor{revisioncolor}{HTML}{246D92}" + "\n"
        + r"\definecolor{baselinecolor}{HTML}{8B6B9F}" + "\n"
        + r"\definecolor{inkcolor}{HTML}{253444}" + "\n"
        + r"\begin{document}" + "\n"
    )
    (target / "figure2_results.tex").write_text(
        preamble + figure + "\n" + r"\end{document}" + "\n", encoding="utf-8")
    with (target / "figure2_source_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print("Figure 2 exported from regenerated evidence (12 plotted rows); no manuscript was edited.")


if __name__ == "__main__":
    main()
