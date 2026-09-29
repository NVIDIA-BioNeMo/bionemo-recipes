# Architecture matching and acceleration depth

Phase 1 decides how the port happens, and in rare cases whether it happens at all. Get this wrong
and everything downstream is a plausible-looking model that silently computes something else.

Most targets are portable. The job of this phase is to pick the right reference to copy and the
right depth to work at — not to look for reasons to refuse.

## Reference implementations you can borrow from

This is what the repo has working, tested TE code for. It is a menu, not a taxonomy: a target that
resembles none of these closely can still be ported, just at less depth and with more of the block
written by hand.

### Encoder / masked LM — pre-norm

**References:** `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py` (canonical), `$BIONEMO_RECIPES/models/codonfm/modeling_codonfm_te.py`,
`$BIONEMO_RECIPES/models/amplify/`
**Converter reference:** `$BIONEMO_RECIPES/models/esm2/convert.py`
**Recipe reference:** `$BIONEMO_RECIPES/recipes/esm2_native_te/`

Fingerprint:

- Bidirectional attention — no causal mask anywhere in the attention path.
- MLM head (`lm_head`, tied or untied to the embedding) or a token/sequence classification head.
- LayerNorm (not RMSNorm), applied *before* the sublayer.
- Plain, non-gated MLP. The activation is a config value, not part of the fingerprint — see the
  MLP form row of the rubric.
- Learned absolute, rotary, or no position embeddings.

Built on `te.TransformerLayer`.

### Encoder / masked LM — post-norm

**Reference:** `$BIONEMO_RECIPES/models/geneformer/src/geneformer/modeling_bert_te.py::TEBertLayer`
**Converter reference:** `$BIONEMO_RECIPES/models/geneformer/src/geneformer/convert.py`
(`convert_geneformer_hf_to_te` / `convert_geneformer_te_to_hf`)
**Recipe reference:** `$BIONEMO_RECIPES/recipes/geneformer_native_te_mfsdp_fp8/`

Fingerprint: everything above, except LayerNorm is applied *after* the residual add.

This variant deliberately does **not** use `te.TransformerLayer`, because that layer is pre-norm.
From the `TEBertLayer` docstring:

> Geneformer/HF BERT (POST-norm): Input -> Attention -> Dropout -> Residual Add -> LayerNorm -> MLP
> -> Dropout -> Residual Add -> LayerNorm -> Output. Typical TransformerLayer (PRE-norm): Input ->
> [LayerNorm Attn inside MultiheadAttention] -> Dropout -> Residual Add -> \[LayerNorm MLP inside
> LayerNormMLP\] -> Dropout -> Residual Add -> Output.

So the block is assembled from `te.MultiheadAttention(input_layernorm=False)`, `te.LayerNorm`, and
two `te.Linear`. Geneformer is also an all-ReLU encoder — it *requires* `hidden_act="relu"` — which
makes it the reference to reach for whenever a target's activation or norm placement doesn't look
like ESM-2's.

Note: geneformer has no `BaseModelTest` coverage, so Tier 2 follows the codonfm no-HF-upstream
template described in `references/validation.md`.

### Causal LM, dense

**References:** `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py` (canonical), `$BIONEMO_RECIPES/models/qwen/modeling_qwen2_te.py`,
`$BIONEMO_RECIPES/models/qwen/modeling_qwen3_te.py`
**Converter reference:** `$BIONEMO_RECIPES/models/llama3/convert.py`, `$BIONEMO_RECIPES/models/qwen/convert_qwen2.py`,
`$BIONEMO_RECIPES/models/qwen/convert_qwen3.py`
**Recipe reference:** `$BIONEMO_RECIPES/recipes/llama3_native_te/`

Fingerprint:

- Causal attention mask.
- RMSNorm.
- SwiGLU / gated MLP (`gate_proj` + `up_proj` + `down_proj`).
- Grouped-query attention — `num_key_value_heads < num_attention_heads`.
- Rotary position embeddings.

### Mixture of experts

**References:** `$BIONEMO_RECIPES/models/mixtral/modeling_mixtral_te.py`
**Converter reference:** `$BIONEMO_RECIPES/models/mixtral/convert.py`
**Supporting:** `$BIONEMO_RECIPES/models/mixtral/fused_token_router.py`, `fused_a2a.py`,
`fused_indices_converter.py`

Fingerprint: everything in the dense causal LM, plus a router / `num_local_experts` /
`num_experts_per_tok` top-k gating replacing the dense MLP.

MoE is the hardest port — expert-parallel all-to-all and the fused router are load-bearing. The
Mixtral reference matches straightforward top-k routing over a linear gate. When the routing is
exotic (expert choice, soft MoE, shared experts with unusual normalization), do not force it through
the fused router: port the attention and norm layers, leave the routing code untouched, and say
exactly that in the report.

### Genomics causal LM

**References:** `$BIONEMO_RECIPES/recipes/opengenome2_llama_native_te/` (a Llama-family fork; its
`modeling_llama_te.py` is a managed copy of `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py`)

Fingerprint: the dense causal LM block structure over a nucleotide/codon vocabulary, small vocab,
long context, THD-packed GQA configs. Treat it as a dense causal LM for the rewrite; the distinction
matters for config defaults (vocab padding, sequence length, packing) and for which recipe to copy.

### Encoder–decoder

**Reference:** none in this repo — this is a TE capability entry, not a worked model.

`te.TransformerLayer` supports cross-attention natively via `layer_type`. From the vendored copy at
`$BIONEMO_RECIPES/recipes/codonfm_ptl_te/src/models/components/encodon_te_layer.py`:

> `layer_type: {'encoder', 'decoder'}, default = 'encoder'` — if set to `decoder`, an additional
> cross-attn block is added after self-attn. This can be used for structures like `T5` Transformer
> in conjunction with the `encoder` option.

The same file builds `self.inter_attention` with `attention_type="cross"` when
`layer_type == "decoder"`, and threads `encoder_output` and `enc_dec_attn_mask` through the forward.
Every BioNeMo model happens to pass `layer_type="encoder"`; that is a gap in coverage, not a
limitation of TE.

Build the encoder stack the way the encoder references do, then the decoder stack with
`layer_type="decoder"`, `self_attn_mask_type` causal, and `enc_dec_attn_mask_type` for the cross
path. Because nothing in-repo exercises this, three caveats are mandatory in the report:

- **No converter template.** `$BIONEMO_RECIPES/models/esm2/convert.py::_pack_qkv_weight` packs
  self-attention QKV only. `inter_attention` takes Q from the decoder and KV from the encoder
  output, so that transform is hand-written.
- **No `BaseModelTest` coverage.** The harness assumes a single stack. Tier 2 is the codonfm
  no-HF-upstream path at best; Tier 1 parity is the real proof.
- **THD packing across a cross-attention boundary is untested here.** Default to BSHD.

### Mixing pieces

Nothing requires a target to take everything from one entry. A post-norm encoder with continuous
inputs can take its block from geneformer and its `quantized_model_init` handling, FP8 config
plumbing, and padded-vocab treatment from `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py`. Say in
`.bionemo-accel/match.json` which reference each piece came from.

## The rubric

Score the target against the reference menu above on six axes. Record each as `match` / `mismatch` /
`unknown` in `.bionemo-accel/match.json` with the specific deciding evidence (file and symbol) —
"looks like Llama" is not evidence; `model.py::DecoderBlock` uses `RMSNorm` + `SwiGLU` +
`num_key_value_heads=8` is.

| Axis                | What to look for                                              |
| ------------------- | ------------------------------------------------------------- |
| Attention masking   | causal vs bidirectional; sliding window; any custom bias term |
| Normalization       | LayerNorm vs RMSNorm; pre-norm vs post-norm                   |
| MLP form            | plain up/down + activation vs gated SwiGLU vs MoE             |
| Positional encoding | learned absolute, RoPE, ALiBi, relative, none                 |
| Attention grouping  | MHA vs GQA vs MQA                                             |
| Head structure      | MLM, causal LM, classification, regression, multi-task        |

**All six axes are advisory.** A mismatch is a caveat in the report, never a rejection — it only
picks which reference to mirror and how much of the block is hand-built; match against the
"Fingerprint" lists above for the mechanics of each. Three traps are worth flagging explicitly:

- **Attention masking is a kwarg, not a fork.** `self_attn_mask_type=` / `window_size=` on
  `te.TransformerLayer` cover causal, bidirectional, sliding window, and arbitrary custom masks — see
  the Depth B example in `references/te-conversion.md`. What has no TE analogue is a non-dot-product
  attention *mechanism* (see "No full TE block" below), not a masking difference.
- **Normalization placement picks the depth track, not just the reference.** Pre-norm builds on
  `te.TransformerLayer` (Depth B); post-norm cannot and needs the hand-assembled block (Depth
  B-postnorm, geneformer's `TEBertLayer`) — see "Encoder / masked LM — post-norm" above.
- **Positional encoding can be load-bearing at init time.**
  `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py::NVEsmEmbeddings` raises on anything but rotary —
  a target with no position encoding drops the `RotaryPositionEmbedding` construction entirely rather
  than configuring it off.

## No full TE block: best-effort Depth C

No architecture family is refused outright. Four families have no TE analogue for their top-level
block — but that is not the same as zero TE benefit. Route these to Depth C in
`references/te-conversion.md`: swap what's swappable (linear projections, norms, MLPs, and, where a
bare attention core fits, the attention computation itself), name the part that stays custom, and
validate Tier 1 only.

- **Diffusion / score-based models** — the denoiser conditioning path (timestep embeddings, AdaLN
  modulation) is not expressible as a `te.TransformerLayer`. The linear/norm/MLP layers elsewhere in
  the denoiser still swap.
- **GNNs and equivariant networks** (SE(3), E(3), tensor-field networks) — message passing and
  irrep-typed tensors have no TE analogue. Any plain linear/norm/MLP sub-layers outside the
  message-passing step still swap.
- **State-space models** (Mamba, S4, Hyena) — the sequence-mixing block is a scan, not attention, and
  has no TE analogue. Everything around the scan — input/output projections, norms, MLPs, and any
  interleaved full-attention layers — still swaps. Evo2's Hyena stack in
  `$BIONEMO_RECIPES/recipes/evo2_megatron/src/bionemo/evo2/models/megatron/hyena/hyena_layer_specs.py`
  and `hyena_layer.py` is this repo's own worked example: the mixer (`HyenaMixer`, the actual
  conv/recurrence) is unconditionally custom, but its projections, norm, and MLP — and the
  interleaved attention layers' projections, norm, MLP, and attention core — are all built as
  independently swappable slots and share the same TE building blocks. When the target already
  builds its block from similarly named, swappable sub-modules, mirror that shape; otherwise an
  inline pass over the target's `nn.Linear`/`LayerNorm`/attention calls is a fine substitute — see
  `references/te-conversion.md`.
- **Non-dot-product attention mechanisms** — linear/kernelized attention (Performer, linear
  transformers), retrieval-augmented attention, and block-sparse routing that changes the compute
  pattern itself (not just which positions are masked) have no TE analogue; `DotProductAttention`
  only implements maskable scaled-dot-product attention (masking itself is config — see the rubric
  above). The projections, norms, and MLPs around the attention computation still swap, and where the
  core is genuinely scaled-dot-product attention under nonstandard QKV/mask shaping, it often still
  fits: see `$BIONEMO_RECIPES/recipes/codonfm_ptl_te/src/models/components/encodon_te_mha.py` for
  `DotProductAttention` used standalone, for both self- and cross-attention.

Not a hard stop, but route elsewhere: **Megatron-LM based code** —
`$BIONEMO_RECIPES/recipes/eden_megatron/` and `$BIONEMO_RECIPES/recipes/evo2_megatron/` already
handle precision through `--mixed-precision-recipe`. Point the user there rather than porting.

Not a hard stop, but Tier 1 only: **vision-only backbones** — `$BIONEMO_RECIPES/recipes/vit/` exists
but has no `BaseModelTest` coverage, so parity rests on the Tier 1 check alone. Say so in the report.

## Hard stop: when no port can be attempted at all

Only two conditions block every depth, including Depth C, because they mean the target cannot even
be inspected or run — not because of its architecture. Use failure class `ARCH_`:

- The model definition cannot be located or is generated dynamically at runtime.
- No forward pass can be run for the parity check because of a reason intrinsic to the target —
  no weights, no tokenizer, no sample input — because then Phase 5 cannot prove anything.

Do **not** use `ARCH_` when a forward pass cannot be run because a dependency failed to install.
That is failure class `ENV_`: the target has not been judged, and the run may succeed in a clean
environment. See the "Failure classes" section in `SKILL.md`.

In this case, write the hard-stop variant of `assets/ACCELERATION_REPORT.md.tmpl` and change
nothing else in the target. It must state:

1. What was detected, with the file and class names (or the absence of any) that led here.
2. The **specific reason no depth can be attempted** — the model definition could not be located, or
   no forward pass could be run. An architecture mismatch or advisory-axis mismatch (including
   attention masking) is never the reason; those go to Depth C instead.
3. Which accelerations, if any, would still be safe to apply by hand — for example, TE `FusedAdam`
   and `torch.compile` are architecture-agnostic — clearly marked as *not applied and not validated
   by this skill*.
4. A pointer to the nearest recipe if the user wants to port manually.

Then exit; do not offer to "try anyway".
