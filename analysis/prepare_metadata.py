"""Maintainer helper: prepare provenance and remove case IDs from numeric exports.

Requires the original medic source directory. It does not copy raw questions or
trace contents. Regenerates public metadata; review source changes explicitly.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from verify_release import ROOT, public_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    paired_path = ROOT / "results/reanalysis_paired_results.csv"
    with paired_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [f for f in reader.fieldnames if f != "complete_case_ids"]
        paired = [{f: row[f] for f in fields} for row in reader]
    with paired_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    mapping = {
        "analysis/build_rewrite_evidence_map.py": "scripts/build_rewrite_evidence_map.py",
        "analysis/reanalyze_rewrite.py": "scripts/reanalyze_rewrite.py",
        "analysis/audit_reanalysis.py": "manuscript_jbi_english/scripts/audit_manuscript.py",
        "analysis/check_tie_policy.py": "manuscript_jbi_english/scripts/check_tie_policy.py",
        "analysis/build_figures.py": "manuscript_jbi_english/scripts/build_figures.py",
        "historical/conditions.py": "scripts/run_propagation_experiment.py",
        "tests/test_reanalyze_rewrite.py": "scripts/test_reanalyze_rewrite.py",
        "tests/test_propagation.py": "tests/test_propagation.py",
        "protocol/retrospective_reanalysis_protocol_original.md": "paper/rewrite/回顾性重分析定义与处理协议_v1.md",
        "metadata/archived_reanalysis_input_sha256.csv": "paper/rewrite/evidence/reanalysis_input_sha256.csv",
    }
    for path in (ROOT / "historical/seal_med").glob("*.py"):
        if path.name != "__init__.py":
            mapping[path.relative_to(ROOT).as_posix()] = "seal_med/" + path.name
    for path in (ROOT / "results").glob("*"):
        mapping[path.relative_to(ROOT).as_posix()] = (
            "manuscript_jbi_english/figures/" + path.name if path.suffix == ".tex" or path.name == "figure2_source_data.csv"
            else "paper/rewrite/evidence/" + path.name)
    rows = []
    target = ROOT / "metadata/source_manifest.json"
    for path in sorted(public_files()):
        if path == target:
            continue
        name = path.relative_to(ROOT).as_posix()
        row = dict(release_path=name, release_sha256=digest(path), bytes=path.stat().st_size)
        original = mapping.get(name)
        if original:
            original_path = source / original
            row.update(source_path="medic/" + original, source_sha256=digest(original_path),
                       status="byte_identical_copy" if digest(path) == digest(original_path) else "documented_release_adaptation")
        else:
            row["status"] = "new_release_documentation_or_tool"
        rows.append(row)
    manuscript = source / "manuscript_jbi_english/main.tex"
    document = dict(schema="focused_paper_code_release_v1", manuscript_title=
        "Disentangling peer revision and vote-order effects in medical multi-agent decision making: a retrospective trace analysis",
        manuscript_source_sha256=digest(manuscript), new_model_calls=0,
        source_root_label="medic", files=rows)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared source manifest for {len(rows) + 1} public files; paired export omits case identifiers.")


if __name__ == "__main__":
    main()
