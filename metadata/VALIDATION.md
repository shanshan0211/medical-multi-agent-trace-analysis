# Release validation

Validation completed locally on 11 October 2026 with Python 3.12. All original workspace files were read-only during packaging; all reanalysis outputs were generated in a separate temporary copy of the release. No model/network calls were made.

| Check | Result |
| --- | --- |
| Code-only unittest run without raw data | 31 passed; one retained-trace regression explicitly skipped |
| Synthetic demo | Source-only tie, post-exposure revision, baseline agreement, and hidden-source voting examples completed |
| Exported numerical summaries | All 48 arm rows and 76 paired comparisons verified |
| Archived input staging | 26 traces, four reference CSVs, one case-ID file, and original protocol hashes matched |
| Full offline reanalysis | 1,549 completed records including the one drill; 2,400 peer rows; 22 design-violation records; zero audit issues |
| Independent reconstruction | 1,548 research records, 48 arm rows, and 76 paired rows verified |
| Export versus fresh full reanalysis | Every exported field matched, after the documented local-input path relocation; all 12 plotted data rows matched |
| Full-data unittest run | All 32 tests passed, no skips |
| Fixed-ballot policy replay | DeepSeek first/last/majority: 14/8/8; GLM: 15/6/6 |
| Figure source export | Standalone Figure 2 TikZ and source data generated successfully; no manuscript edited |
| Public-file review | No raw trace files, benchmark question text, credential files, or clinical-review documents in the allowlist |

The release verifier checks the final allowlist and SHA-256 values and scans for common API/GitHub/AWS token formats, credential assignments, and private-key headers. It found no such matches. This is a limited automated check, not a proof that every possible sensitive string is absent.

The original research snapshot verified 85 files. The focused staging snapshot verifies 32 inputs; the independent adapted audit checks 34 distinct input/code hashes. This reduced scope is documented in `ADAPTATIONS.md`; it is not a change to the scientific denominator or an assertion of preregistration.

The validation copy is disposable and reproducible with the README commands. Raw archived inputs remain in the original workspace and are not part of the GitHub upload folder.
