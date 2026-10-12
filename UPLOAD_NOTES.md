# GitHub handoff

Only this focused release folder is in scope for upload. Do not upload its parent research workspace.

The package currently contains selected code, synthetic tests, aggregate numeric results, figure sources, protocol documentation, and checksums. It excludes raw inputs, API credentials, model-call transports, clinical-review documents, old paper drafts, build artifacts, and vendored external repositories.

If full reproduction has been run inside this folder, `local_inputs/`, `paper/`, and `generated_figures/` contain local material and are ignored by Git. The release verifier ignores them deliberately and checks the public allowlist. **A browser folder upload or ZIP created from all files does not honor `.gitignore`.** Use only the files in `metadata/source_manifest.json`, or stage with Git while respecting `.gitignore`; review the staged filenames before pushing.

Before a public upload, the authors need to choose the destination repository, visibility, and a code license (or explicitly retain the unlicensed status). Benchmark redistribution is a separate decision. No GitHub repository has been created and no remote changes have been made.

Suggested repository description: Offline trace reconstruction and voting sensitivity analysis for medical multi-agent decision making.
