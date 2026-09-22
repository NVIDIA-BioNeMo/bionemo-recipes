# `automodel/`

NeMo AutoModel integration: the seams that let a host LM carry an encoder,
projector, and soft tokens while training through AutoModel's recipe,
checkpoint, and parallelism machinery.

- `model.py` — the host contract: `MultimodalInputMixin` (flat forward kwargs,
  embedding-boundary scatter bridge), projector construction in either
  ownership mode, and `build_multimodal_host`.
- `registry.py` — idempotent out-of-tree architecture registration.
- `checkpoint.py` — DTensor-safe parameter copies and the state-dict adapter
  that keeps projector tensors out of HF export/refit.
- `recipe.py` — `ProjectorRecipeMixin`: topology guards, projector
  materialization, sidecar checkpoint state, artifact export.
- `optim.py` — `ProjectorAdamWConfig`: trainability applied at the last safe
  point (after PEFT freezing, before optimizer construction).
- `parallel.py` — replicated-projector protocol under TP, and TP×EP
  composition for module-owned projectors.
- `data.py` — manifest-backed feature datasets and chat collation, including a
  bounded-memory iterable packer for AutoModel's typed dataloader path.

`build_multimodal_host(..., additional_parameter_prefixes=(...))` declares
consumer-owned EXTRA parameters absent from the base checkpoint. The native
state-dict adapter excludes them from strict base loading and HF refit export
under either projector ownership mode. This is also the frozen GRPO contract
for learned token embeddings loaded into projected vLLM once from the
projector artifact.

## Upstream gaps

Subset of `docs/upstream-gaps.md`
(the single source of truth). This subpackage's modality-roadmap items are U-2,
U-7, and U-10; U-11 and U-13 through U-17 are recorded there as adjacent work.
