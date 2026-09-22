# `vllm/`

vLLM integration: `build_mm_plugin` (`plugin.py`), one factory for out-of-tree
projected-token modalities.

- `mode="projected"` serves pre-projected `[T, H_lm]` soft tokens straight to
  `embed_multimodal` — vLLM never loads the modality encoder.
- `mode="encode"` builds the processing layer (data parser, processor, prompt
  replacement, dummy inputs) around a raw encoder payload.

Projected mode can also replace selected vocabulary-token embeddings with
frozen vectors from the projector artifact's EXTRA state:

```python
register_vllm = build_mm_plugin(
    # ...the ordinary projected-mode arguments...
    token_embedding_overrides={
        "<SPECIAL_START>": "learned_start_embedding",
        42: "learned_end_embedding",
    },
)
```

The vLLM HF config must carry `mm_projector_artifact_path` (or the attribute
named by `embedding_override_artifact_attr`). Each string must tokenize to
exactly one token, every ID must be in vocabulary, and each declared EXTRA
tensor must be `[H_lm]`. Only declared keys are used; unrelated EXTRA state is
ignored. Overrides are applied after vLLM's normal multimodal merge, so the
projected placeholder scatter is preserved.

The vectors are loaded once as non-persistent frozen buffers. For GRPO, pass
the same EXTRA tensor names as `additional_parameter_prefixes` to
`build_multimodal_host`; its state-dict adapter then keeps them out of repeated
policy-to-vLLM refits while the backbone and merged LoRA weights continue to
refit normally.

vLLM 0.25's public model and multimodal-processor registries are used
unchanged; consumers supply modality dimensions and token strings through
configuration. All vLLM imports stay inside the factory so the base wheel
imports without vLLM installed.

## Upstream gaps

Subset of [`docs/upstream-gaps.md`](../../../docs/upstream-gaps.md)
(the single source of truth). U-22 through U-24 are recorded there as intended
model/processor extension code, not upstream asks.
