# Repository Instructions

- This Git repository is the single maintained source for Flash-LIBERO. Work from the current branch/working tree; do not overwrite it with an older owner snapshot or archived implementation.
- Preserve existing uncommitted changes and evidence. Do not reset/clean a workspace or create worktrees unless specifically authorized.
- Preserve the original `libero.libero` public API and canonical `libero.gpu` namespace. GPU is the existing default backend.
- Keep adopted algorithms, explicit entry points, domain selectors, source admission, capacity checks, owner replacement/restoration, errors, and fallback paths together.
- Existing accepted GPU correctness and floating-point error bounds remain accepted. Workspace, installation, or packaging work checks integration completeness without adding full-application numerical or state-byte gates.
- For newly modified mathematics, use the independent function's floating-point error, residual, and sensitivity bounds. Do not require all floating-point intermediates to be bitwise identical. Indexing, stable order, copy bit patterns, capacity, owner, error, and public hold contracts remain exact. Legal Newton stopping/line-search choices may change iteration counts.
- Measure local performance in the relevant real input domains. Keep positive, conditional, mixed, reverted, excluded, and candidate outcomes distinct. Do not globally enable every positive prototype or combine unlike timing baselines.
- Serialize GPU/nvcc/LTO work on a shared device. Preserve build/version bindings and explicit invalidation; source caches must not bypass admission.
- Public documentation is English. Keep third-party licensing and source attribution intact.
- Use `python tools/check_source.py` for source checks, `--cpu-tests` for the supported light suite, and `python tools/release/build.py` for checked release artifacts. No repeated build is required after successful checks unless a new change or failure justifies it.
