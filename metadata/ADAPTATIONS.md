# Release adaptations

The SHA-256 manifest records both source and released files. Byte-identical copies are distinguished from adapted files. All original workspace files remain unchanged.

| Released file | Adaptation |
| --- | --- |
| `analysis/build_rewrite_evidence_map.py` | Keep only the helper functions used by the reanalysis; remove broad inventory generation; use ignored `local_inputs/` paths. Normalization, reference mapping, and trace parsing are unchanged. |
| `analysis/reanalyze_rewrite.py` | Relocate protocol and helper paths. Preserve all endpoint, partition, last-record selection, comparison, and statistical calculations. |
| `analysis/audit_reanalysis.py` | Relocate paths; preserve independent scientific reconstruction; remove author-facing clinical-review and manuscript-report writers. |
| `analysis/check_tie_policy.py` | Relocate evidence path; preserve exact-label fixed-ballot replay. |
| `analysis/build_figures.py` | Preserve result-figure calculation; export a standalone TikZ figure instead of editing manuscript blocks. The already prepared Figure 1 is distributed separately. |
| `historical/conditions.py` | Extract role and condition builders from the historical runner, without its network, credential, case-loader, or execution CLI. Unused compatibility conditions are not claims about paper evidence. |
| `historical/seal_med/__init__.py` | Minimal package entry point instead of importing unrelated SEAL-Med pipelines. The other listed implementation modules are unmodified. |
| Tests | Relocate imports; explicitly skip the retained-trace regression when no generated full case table is present. Synthetic assertions remain unchanged. |
| `results/reanalysis_paired_results.csv` | Remove only the `complete_case_ids` column; retain all 76 comparisons and every numeric statistic. |

Staging produces a new 32-entry snapshot (31 required data files and the unchanged original protocol). It is a packaging integrity check, not the original broader 85-entry workspace snapshot. The original 34-entry reanalysis manifest is preserved separately for source verification. Release scripts and original protocol are separately hashed by the rerun; hash counts and paths therefore differ from the archived summary, while scientific outcomes must match.

The original summary in `results/reanalysis_summary.json` remains byte-identical and describes the original audit's 85 input checks. It does not claim that this reduced release includes those 85 files. Raw data and author-review documents are intentionally excluded from the public allowlist.
