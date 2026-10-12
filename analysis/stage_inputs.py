"""Copy the exact authorized archived inputs into an ignored local directory."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Archived medic directory, or authorized copy with identical paths")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    with (ROOT / "metadata/archived_reanalysis_input_sha256.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    selected = [e for e in entries if e["path"].startswith(("outputs/propagation/", "data/curated_cases/"))]
    if len(selected) != 31:
        raise RuntimeError("Expected exactly 31 archived data inputs")
    protocol = next(e for e in entries if e["path"].startswith("paper/rewrite/"))
    protocol_path = ROOT / "protocol/retrospective_reanalysis_protocol_original.md"
    if sha(protocol_path) != protocol["sha256"]:
        raise RuntimeError("Archived protocol changed")
    prepared = []
    for entry in selected:
        source = (source_root / entry["path"]).resolve()
        source.relative_to(source_root)
        target = ROOT / "local_inputs" / entry["path"]
        if not source.is_file() or sha(source) != entry["sha256"]:
            raise RuntimeError(f"Missing or changed archived input: {entry['path']}")
        if target.exists() and (not target.is_file() or sha(target) != entry["sha256"]):
            raise RuntimeError(f"Refusing to replace differing local input: {target.relative_to(ROOT)}")
        prepared.append((entry, source, target))
    trace_dir = ROOT / "local_inputs/outputs/propagation"
    expected = {target.name for _, _, target in prepared if target.suffix == ".jsonl"}
    extra = {p.name for p in trace_dir.glob("*.jsonl")} - expected
    if extra:
        raise RuntimeError(f"Unexpected local trace files: {sorted(extra)}")
    data_dir = ROOT / "local_inputs/data/curated_cases"
    expected_csvs = {target.name for _, _, target in prepared if target.suffix == ".csv"}
    extra_csvs = {p.name for p in data_dir.glob("*.csv")} - expected_csvs
    if extra_csvs:
        raise RuntimeError(f"Unexpected local reference files: {sorted(extra_csvs)}")
    snapshot = []
    for entry, source, target in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != entry["sha256"]:
            raise RuntimeError("Copy integrity check failed")
        snapshot.append(dict(path=target.relative_to(ROOT).as_posix(), sha256=entry["sha256"]))
    snapshot.append(dict(path=protocol_path.relative_to(ROOT).as_posix(), sha256=protocol["sha256"]))
    out = ROOT / "paper/rewrite/evidence"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "source_snapshot_sha256.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256"])
        writer.writeheader()
        writer.writerows(snapshot)
    print("Verified and staged 31 archived data files; packaging snapshot has 32 entries including protocol.")


if __name__ == "__main__":
    main()
