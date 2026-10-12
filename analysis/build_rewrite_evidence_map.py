"""Trace input helpers extracted from the historical inventory script.

The broad exploratory inventory generator is deliberately excluded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "rewrite" / "evidence"
TRACE_DIR = ROOT / "local_inputs" / "outputs" / "propagation"
DATA_DIR = ROOT / "local_inputs" / "data" / "curated_cases"


def rel(path):
    return path.relative_to(ROOT).as_posix()


def key(label):
    # Same normalization as propagation._same_label; no package imports.
    return " ".join(re.sub(r"[^\w]+", " ", str(label or "").casefold()).split())


def same(a, b):
    return bool(key(a)) and key(a) == key(b)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path):
    records, parse_errors, non_objects = [], [], 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                parse_errors.append(line_no)
                continue
            if not isinstance(row, dict):
                non_objects += 1
                continue
            records.append((line_no, row))
    return records, parse_errors, non_objects


def successful(record):
    return record.get("run_status") != "failed" and isinstance(record.get("propagation_metrics"), dict)


def family(name):
    if name.startswith("ext150exp_dsflash_"):
        return "MED_EXT91_DS"
    if name.startswith("ext150exp_glmflash_"):
        return "MED_EXT91_GLM_PARTIAL"
    if name.startswith("ext60_dsflash_"):
        return "MED_EXT59_DS"
    if name.startswith("ext60_glm53flash_"):
        return "MED_EXT59_GLM"
    if name.startswith("holdout60_"):
        if "glm52_thinking" in name:
            return "ARENA60_GLM52_THINKING"
        if "glm52" in name:
            return "ARENA60_GLM52"
        if "v4pro" in name:
            return "ARENA60_DS_PRO"
        return "ARENA60_DS_FLASH"
    return "FORMAT_FAILOVER_DRILL"


def canonical_file(name):
    if name.startswith("ext150exp_"):
        return DATA_DIR / "medxpertqa_text_ext150_expansion91.csv"
    if name.startswith("ext60_"):
        # Full 60 covers the GLM single arm, including Text-2105.
        return DATA_DIR / "medxpertqa_text_test60_external.csv"
    if name.startswith("holdout60_"):
        return DATA_DIR / "diagnosis_arena_closed.csv"
    return DATA_DIR / "medxpertqa_text_test60_external_n59.csv"


def canonical_gold(row):
    options = json.loads(row.get("options_json") or "[]")
    answer = (row.get("answer") or "").strip()
    if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and answer not in options:
        index = ord(answer.upper()) - ord("A")
        if index < len(options):
            answer = options[index]
    return answer, options
