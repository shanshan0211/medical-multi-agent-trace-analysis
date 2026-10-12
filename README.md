# Medical multi-agent trace analysis

Code accompanying **Disentangling peer revision and vote-order effects in medical multi-agent decision making: a retrospective trace analysis** (current English manuscript, prepared for submission to the *Journal of Biomedical Informatics*).

This is a focused extraction from the research workspace, not a dump of all historical experiments. The paper analyzes archived experiments retrospectively. The release performs **no model or network calls**. Numerical exports are from the current reanalysis, not the older stored propagation metrics.

## Start here

Python **3.10 or newer** is required; the offline workflow uses only the standard library. Run commands from this folder. Replace `python` with the path to your Python executable when needed.

```shell
python -m unittest discover -s tests -v
python analysis/demo.py
python analysis/check_summary_statistics.py
python analysis/verify_release.py
```

The first command tests observable pathways, voting, gating, and statistical invariants with synthetic inputs. The retained-trace regression is explicitly skipped until the full local reanalysis has been generated. The demo illustrates a source-only tie and a peer revision; it is not a new experiment. The summary check recomputes Wilson intervals, paired intervals, exact McNemar tests, and global Holm adjustment from the exported counts. It does not claim to reconstruct raw traces without data.

## Folder guide

| Folder | Purpose |
| --- | --- |
| `analysis/` | Offline reanalysis, independent reconstruction, tie replay, result-figure generation, and release checks |
| `historical/` | Selected propagation implementation and its import dependencies; injectable model boundary, no network client |
| `tests/` | Synthetic mechanism/statistical tests and data-dependent regression |
| `results/` | 48 arm-summary rows, 76 paired comparisons, Figure 2 source data, and vector figure sources |
| `protocol/` | Original retrospective protocol plus English operational definitions |
| `metadata/` | Original input hashes and source-to-release provenance |

`historical/` preserves current source code for protocol inspection. Its optional legacy features and unused condition builders are retained for compatibility, **not additional paper evidence**. The authoritative evidence scope is the 26 trace files listed in `metadata/archived_reanalysis_input_sha256.csv`: 25 research traces and one engineering drill. The current implementation is not proof of the exact historical vendor checkpoint or all historical request parameters.

## Full local reproduction with archived inputs

Raw benchmark questions, answer strings, model traces, case-level tables, and clinical-review packets are deliberately absent from the distribution. Redistribution permissions for third-party benchmarks have not been confirmed. The original workspace already contains the required inputs. Supply that workspace's `medic` directory, or an authorized copy with the same relative paths:

```shell
python analysis/stage_inputs.py --source-root "PATH/TO/medic"
python analysis/reanalyze_rewrite.py
python analysis/audit_reanalysis.py
python analysis/check_tie_policy.py
python -m unittest discover -s tests -v
python analysis/check_summary_statistics.py --compare-generated
python analysis/build_figures.py
```

The staging script verifies SHA-256 values against the archived manifest **before copying** exactly 26 JSONL traces, four curated CSVs, and one case-ID file. It also checks the unchanged protocol. It fails on missing or changed inputs rather than silently substituting newer data. Inputs go to ignored `local_inputs/`; regenerated evidence goes to ignored `paper/rewrite/evidence/`; generated figures go to ignored `generated_figures/`. Existing files are reused only when their hashes match.

The release's staging snapshot covers the 31 required data files and original protocol. This is distinct from the original broad 85-file research-workspace snapshot, which covered many unrelated historical files. The reanalysis separately hashes its adapted code. Do not interpret the packaging snapshot as preregistration.

Optional figure compilation requires a local TeX installation with TikZ and the `standalone` class:

```shell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=generated_figures generated_figures/figure2_results.tex
```

## Expected checks

- 1,548 completed research case-configuration records; 10 failed and 26 planned but not run.
- Including the one engineering drill: 1,549 completed records and 2,400 peer records.
- 22 injection-design violation records (11 cases in two arms), kept and identified separately.
- 48 arm-summary rows and 76 paired comparisons with global Holm correction.
- Core ungated post-exposure revisions: DeepSeek 5, GLM 1.
- Fixed-ballot target selections under first/last/strict-majority policies: DeepSeek 14/8/8 and GLM 15/6/6.

See [protocol/analysis_definitions.md](protocol/analysis_definitions.md) for endpoint meanings and interpretation limits. Post-exposure revision is an observed sequence, not demonstrated causal adoption. Fixed-ballot replay changes aggregation only; it does not rerun reasoning. Clean-task comparisons do not establish an accuracy gain.

## Publication status

This folder is prepared for GitHub review; it has **not been uploaded**. A repository name, destination account, and visibility have not yet been chosen. No open-source license has been assigned: see [LICENSE_STATUS.md](LICENSE_STATUS.md). Inspect [UPLOAD_NOTES.md](UPLOAD_NOTES.md) before publication. Credentials, original model-call transports, old manuscripts, unrelated experiments, and third-party repositories are excluded.
