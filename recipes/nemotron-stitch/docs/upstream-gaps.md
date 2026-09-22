# Upstream gaps

This is the tracker for framework seams owned by `nemotron-stitch` and the
upstream changes that could delete them. Identifiers are stable and must not be
reused. A framework-workaround comment cites its row, names the pinned
revision, and states the deletion condition.

The first table is the upstream roadmap: each item removes friction specific
to connecting an encoder, projector, or soft tokens to the frameworks. The
second table preserves useful findings that are general framework work,
model-specific support, intended extension code, or local cleanup; those rows
are explicitly not Stitch's upstream roadmap.

Statuses were checked against the
[upstream multimodal fixes sheet](https://docs.google.com/spreadsheets/d/1nvIdcJcqJpK-GJmhxRzDuwjqbz5CkZJYHFzEXctcS3E/edit?gid=0#gid=0),
GitHub, and upstream main on 2026-09-16 (NeMo RL adoption pin `4d969c93`;
AutoModel remains locked at `1814c6c9`; vLLM `cee0f92c`).

## Modality-integration roadmap

| ID                                                        | Status                                                                                                                                                                                   | Local seam                                                                                              | Upstream value and deletion condition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **U-1** NeMo RL tokenizer/processor kwargs                | **Adopted.** [RL#3798](https://github.com/NVIDIA-NeMo/RL/pull/3798), merge `a8d4c4e6`.                                                                                                   | None.                                                                                                   | Accepted as a small, general configuration passthrough; the local tokenizer patch is deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **U-2** AutoModel out-of-tree registration                | **Merged; adoption blocked on NeMo RL's AutoModel lock (pin `4d969c93` still pins `1814c6c9`).** [Automodel#3645](https://github.com/NVIDIA-NeMo/Automodel/pull/3645), merge `60790465`. | `register_models` (`automodel/registry.py`).                                                            | Accepted public registration and entry-point discovery. Adopt it and delete the private registry import when the lock advances.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **U-4** NeMo RL modality-neutral vLLM prompt data         | **Adopted.** [RL#3803](https://github.com/NVIDIA-NeMo/RL/pull/3803), merge `4d969c93`.                                                                                                   | None.                                                                                                   | NeMo RL now preserves its modality-neutral `vllm_multi_modal_data` mapping through collation, rollout, formatting, and generation-only evaluation. The local prompt formatter replacement, packed rollout route, and encoder eval collator are deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **U-5** NeMo RL worker-extension FQNs                     | **Adopted.** [RL#3809](https://github.com/NVIDIA-NeMo/RL/pull/3809), merge `3c17147e`.                                                                                                   | Worker runtime registration (`nemo_rl/runner.py`).                                                      | Merged as config-driven passthroughs: `policy.worker_extension_cls_fqn` and `generation.worker_extension_cls_fqn`, validated before worker allocation and mutually exclusive with `quant_cfg`. The `grpo.Policy` patch and both resolver patches are deleted; the configs name the package workers and the bootstrap keeps only the `ACTOR_ENVIRONMENT_REGISTRY` runtime registration upstream requires to pre-exist. The resolver seam had also served pre-#3809 pins (genome-research's `81aa43dd`); consumers on such pins adopt the config keys with their pin bump.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **U-7** AutoModel projector trainability                  | **Merged; adoption blocked on NeMo RL's AutoModel lock (pin `4d969c93` still pins `1814c6c9`).** [Automodel#3681](https://github.com/NVIDIA-NeMo/Automodel/pull/3681), merge `506663df`. | `ProjectorAdamWConfig` and its recipe wiring.                                                           | Accepted as generic `freeze_config` selectors rather than a new lifecycle hook. Express the policy as selectors and delete the seams when the lock advances.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **U-8** NeMo RL compact-LoRA warm start                   | **Adopted.** [RL#3874](https://github.com/NVIDIA-NeMo/RL/pull/3874), merge `a0a4b901`.                                                                                                   | Donor provenance gate (`nemo_rl/policy.py`).                                                            | Adopted as `dtensor_cfg.lora_cfg.restore_from`: NeMo RL resolves the compact PEFT export, routes canonical HF keys through the model-family state-dict adapter, validates donor rank/alpha/base-model, and loads before the KL reference capture. The tensor-copying loader is deleted; what remains is the package's fuller provenance check (`initial_adapter_expected_config`) over `adapter_config.json`, run at construction because the upstream PEFT load is non-strict. GroupedExpertsLoRA restore now follows AutoModel's own PEFT loader and still needs a GPU qualification (under the deleted local loader, a qualified Lightning-30B adapter had 92 unknown and 92 missing tensors).                                                                                                                                                                                                                                                                                                                                                                                          |
| **U-10** AutoModel external-tensor checkpoint exclusion   | **Recheck after the AutoModel pin bump; do not propose as currently written.**                                                                                                           | Dynamic `ProjectorStateDictAdapterMixin` composition (`automodel/model.py`, `automodel/checkpoint.py`). | Potentially valuable, but [current main](https://github.com/NVIDIA-NeMo/Automodel/blob/ac44d92faef7d9c1735f1cb98acbdead27b8b181/nemo_automodel/components/checkpoint/config.py) already has public `checkpoint.skip_task_head_prefixes_for_base_model` for base loads. First adopt that and measure what remains. Only propose a small, declarative exclusion for consolidated export/refit if the adapter is still necessary; do not ask for another lifecycle hook.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **U-18** NeMo RL generation-only HF overrides             | **Open.** Re-verified against pin `4d969c93` on 2026-09-16.                                                                                                                              | Final engine-construction override (`nemo_rl/vllm_worker.py`).                                          | `generation.vllm_kwargs.hf_overrides` is accepted and [composed with FP8 quantization without clobbering user values](https://github.com/NVIDIA-NeMo/RL/blob/4d969c93268fda1687fed8ca38668da4087ce76b/nemo_rl/models/generation/vllm/vllm_worker.py), but GRPO still replaces it wholesale from `policy.hf_config_overrides` ([`nemo_rl/algorithms/grpo.py` at `4d969c93`](https://github.com/NVIDIA-NeMo/RL/blob/4d969c93268fda1687fed8ca38668da4087ce76b/nemo_rl/algorithms/grpo.py), "make vllm hf overrides match the training policy"), and the training policy must name the AutoModel host class where the engine needs the plugin-registered rollout class. The override translates `architectures` at engine construction and fails closed on every other conflict. Delete it when GRPO merges user `hf_overrides` instead of replacing them; an upstream PR is warranted.                                                                                                                                                                                                        |
| **U-26** AutoModel evaluation-only recipe entry point     | **Open.**                                                                                                                                                                                | `ProjectorEvaluationRecipe` (`automodel/recipe.py`).                                                    | AutoModel `1814c6c93a66b9d59d254960ef6a99a64249b671` has no public evaluation-only entry point for a configured fine-tuning recipe: setup requires an otherwise-unused train dataset, train loader, optimizer, and scheduler, while running validation without an update requires private `_run_validation_epoch`, metric-logger, and checkpointer lifecycle methods. Stitch owns one shared adapter so modality examples do not copy that framework-sensitive code. Delete it and the examples' dummy setup loaders when AutoModel provides a public mode that restores the normal model/artifact path, iterates only validation loaders, reports metrics, and closes framework resources.                                                                                                                                                                                                                                                                                                                                                                                                |
| **U-27** AutoModel serve-ready merged PEFT export         | **Open (not worked around).** [Automodel#3834](https://github.com/NVIDIA-NeMo/Automodel/pull/3834).                                                                                      | None.                                                                                                   | AutoModel `1814c6c93a66b9d59d254960ef6a99a64249b671` deliberately disables its normal consolidated-HF export for PEFT checkpoints. A private EAGLE-recipe helper merges ordinary `LinearLoRA`, but there is no public model-family-aware export that also folds `GroupedExpertsLoRA` into the canonical HF expert tensors through the model's state-dict adapter. A concrete BioReason-RNA reproduction on 2026-09-06 used AutoModel's `tools/merge_lora.py`, PEFT 0.18.1, Transformers 5.12.1, the Lightning-30B base, and the completed rank-192/alpha-384 broad adapter from step 19,999; loading failed before merge because PEFT constructed each 128-expert fused rank dimension as `128 * 192 = 24,576` while the checkpoint correctly stored AutoModel's MoE-rank-scaled `128 * 32 = 4,096` dimension. Expose a tested operation in AutoModel that restores the AutoModel PEFT-v5 checkpoint through the model's state-dict adapter and emits canonical sharded HF weights; do not copy the LLaVA example's dense-only merge script or an expert-layout rewrite into applications. |
| **U-28** NeMo RL variable-length payload dynamic batching | **Resolved locally; no upstream change needed.**                                                                                                                                         | Flat encoder and rollout tensors per logical row (`[N, H]`).                                            | NeMo RL main `71a08314` already retains each `PackedTensor` as one logical row through GRPO repetition, deduplication, selection, and dynamic-microbatch slicing. Wrapping the canonical flat tensor directly lets its existing materialization concatenate along `N`, producing `[N₁ + N₂, H]`; Stitch then derives row-major projector positions from `[B, S]` `input_ids`. A pinned-framework regression covers unequal rows and projected rollout extraction; a GB300 probe with BioReason-RNA cache rows `[50, 4096]` and `[240, 4096]` materialized `[290, 4096]` with exact content preservation and scattered all 290 projected rows at the derived positions.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-29** NeMo RL grouped-expert LoRA refit                | **Open (not worked around).**                                                                                                                                                            | None.                                                                                                   | NeMo RL main `71a08314`'s DTensor-v2 refit generator merges only `LinearLoRA` modules whose base tensor ends in `.weight`; AutoModel `1814c6c9` represents routed-expert LoRA as `GroupedExpertsLoRA`, so the grouped base tensors remain unchanged and the four raw `lora_gate_and_up_*` / `lora_down_*` parameters are sent as additional weights. A container probe with nonzero deltas reproduced both behaviors, so vLLM cannot receive the policy represented by that training graph. The upstream refit exporter must fold every supported AutoModel LoRA representation into its canonical HF base tensors (or provide an equivalent adapter-aware transfer contract) before this target family can be used in GRPO. Do not add an application merge/conversion shim.                                                                                                                                                                                                                                                                                                              |
| **U-31** AutoModel `logits_to_keep` bridge                | **Open.**                                                                                                                                                                                | Explicit `logits_to_keep` bridge (`automodel/model.py`).                                                | AutoModel `7d36972d` downgrades configured `FusedLinearCrossEntropy` when a decorated host accepts `logits_to_keep` through `**kwargs` rather than naming it explicitly. Delete the bridge when loss capability detection accepts forwarded kwargs or a model capability marker.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **U-35** NeMo RL encoder-vLLM expert parallelism          | **Resolved locally; no upstream change needed.**                                                                                                                                         | Encoder generation topology gate (`nemo_rl/vllm_worker.py`).                                            | NeMo RL `4d969c93268fda1687fed8ca38668da4087ce76b` passes `expert_parallel_size` through its public vLLM configuration and maps the qualified `EP=TP` topology to vLLM's expert-parallel engine without additional data-parallel actors. Stitch admits `EP=TP` while continuing to reject unqualified pipeline parallelism and `EP!=TP`; a BioReason-RNA 8×H100 generation probe qualifies `TP=EP=8`. No framework workaround or upstream proposal is required.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **U-43** Full decoder SFT with projector trainability     | Implemented locally; upstream adoption tracked by U-7.                                                                                                                                   | Explicit decoder selectors in `ProjectorFinetuneRecipe` / `configure_trainable_parameters`.             | Preserves native decoder freezes and independent projector trainability; rejects missing selectors and LoRA/full ambiguity. Delete the selection seam after adopting AutoModel freeze selectors (U-7). See historical reproduction and qualification below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Recorded, but not Stitch upstream work

| ID                                                          | Classification                                                         | Maintainer view                                                                                                                                                                                                                                                                                                                                                                                                         | Repository action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **U-3** AutoModel dataloader shuffle                        | Resolved general bug.                                                  | Already fixed by [Automodel#2390](https://github.com/NVIDIA-NeMo/Automodel/pull/2390) / `c4025978`.                                                                                                                                                                                                                                                                                                                     | Keep only as identifier history; local copy is deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **U-6** NeMo RL load format                                 | Withdrawn proposal.                                                    | Upstream's model-aware `auto` behavior is the correctness boundary; [RL#3810](https://github.com/NVIDIA-NeMo/RL/pull/3810) closed unmerged.                                                                                                                                                                                                                                                                             | Keep only as identifier history; local bypass is deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **U-9** NeMo RL colocated residency                         | General performance policy.                                            | Potentially useful, but risky cache/offload semantics and hardware-dependent value demand broad benchmarks. It should not ride a modality argument.                                                                                                                                                                                                                                                                     | Keep the opt-in local seam; pursue upstream only with a generic proposal and evidence. On NeMo RL main `71a08314` with vLLM 0.25.1 the sleep transition is all-or-nothing: a one-GB300 RNA probe improved rollout throughput 22.7% with a 24 GiB KV cache, but retaining that cache through policy training left about 1 GiB of HBM. A supported phase policy should be able to retain rollout weights while discarding and recreating only KV cache.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-11** AutoModel custom-module initialization             | Intended model-extension responsibility.                               | An out-of-tree architecture is expected to initialize the modules it adds. A framework API is unlikely to remove the model-owned `initialize_weights` method.                                                                                                                                                                                                                                                           | Keep the narrow implementation; revisit only if U-2 adoption exposes a reproducible framework bug.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **U-12** AutoModel config composition                       | Unrelated framework ergonomics.                                        | Includes/interpolation may be useful generally, but modality support is not the argument and Stitch owns no implementation.                                                                                                                                                                                                                                                                                             | Do not track as an upstream ask; retain the example's agreement test.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-13** AutoModel feature packing                          | Package contract using a public seam.                                  | [Current AutoModel exposes `PackingConfig.build`](https://github.com/NVIDIA-NeMo/Automodel/blob/ac44d92faef7d9c1735f1cb98acbdead27b8b181/nemo_automodel/components/datasets/loader.py); arbitrary tensor merging and index rebasing are application packing policy, not behavior its stock packers should guess.                                                                                                        | Keep the custom packer; upstream only a concrete reusable primitive discovered while simplifying it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **U-14** AutoModel dense Nano construction                  | Resolved model-support bug.                                            | Fixed by [Automodel#2670](https://github.com/NVIDIA-NeMo/Automodel/pull/2670), commit `33052a2b`.                                                                                                                                                                                                                                                                                                                       | Adopt native construction when the NeMo RL lock contains the fix, then delete the Hub fallback.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **U-15** AutoModel buffer placement                         | General correctness bug.                                               | Maintainers should value a minimal reproducer and fix if current main still fails, but it is not modality work.                                                                                                                                                                                                                                                                                                         | Re-test after the pin bump; file a focused bug only if it reproduces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-16** AutoModel TE-linear TP                             | Model/distributed compatibility.                                       | Valuable if a supported AutoModel architecture still needs the conversion on current main; not a modality extension API.                                                                                                                                                                                                                                                                                                | Re-test current main and upstream through that architecture's TP plan, or keep with the consumer topology.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **U-17** AutoModel TP×EP dispatch                           | General distributed-topology bug.                                      | Likely valuable with an upstream architecture reproducer, but should be fixed in AutoModel's strategy dispatch rather than justified by a projector.                                                                                                                                                                                                                                                                    | Pursue independently of Stitch if current main still reproduces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **U-19** NeMo RL `PackedTensor` row API                     | Resolved by U-4 adoption.                                              | None.                                                                                                                                                                                                                                                                                                                                                                                                                   | The only compatibility branch served the deleted local rollout formatter. Native `vllm_multi_modal_data` carries rollout values without `PackedTensor`, so the branch is deleted rather than updated for another pin.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-20** NeMo RL validation sampling                        | General generation feature.                                            | Distinct validation sampling may be useful, but it is unrelated to modality addition and current main deliberately limits it to NeMo Gym.                                                                                                                                                                                                                                                                               | Keep the config aliases; do not drive this from Stitch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **U-21** NeMo RL per-datum chat-template kwargs             | General data API.                                                      | Plausibly useful for reasoning/tool templates, but the API and security surface need general consumers; modality is not the justification.                                                                                                                                                                                                                                                                              | Keep the small proxy; upstream separately if NeMo RL wants the general capability.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **U-22** vLLM post-merge embeddings                         | Intended model behavior.                                               | [vLLM's contributor guide](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/docs/contributing/model/multimodal.md) explicitly says models may override `embed_input_ids` for additional merge logic.                                                                                                                                                                                  | Keep the override and stop treating it as an upstream gap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **U-23** vLLM custom-modality MRoPE                         | Model-specific position policy.                                        | vLLM cannot infer positional semantics for an arbitrary modality. A new public hook is unlikely to beat the current model override without more consumers.                                                                                                                                                                                                                                                              | Keep the narrow override; open an issue only if a generic policy emerges.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **U-24** vLLM processor methods                             | Documented extension surface.                                          | [vLLM's contributor guide](https://github.com/vllm-project/vllm/blob/cee0f92c02112ace7120da45896d27e0fe95ea1e/docs/contributing/model/multimodal.md) explicitly requires subclasses to implement `_get_mm_fields_config` and `_get_prompt_updates`; the underscore names do not make their use a workaround.                                                                                                            | Keep the processor subclass and stop treating it as an upstream gap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **U-25** NeMo RL grouped/3D-expert refit                    | Monitored external work.                                               | [RL#3651](https://github.com/NVIDIA-NeMo/RL/pull/3651) merged upstream (`a366bc8c`, 2026-09-06): refit through vLLM's `reload_weights` API. Not load-bearing for the qualified colocated DTensor-policy path.                                                                                                                                                                                                           | Identifier history; reassess only if the example adopts that refit stream.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **U-30** AutoModel Mamba `out_proj` LoRA                    | Model-specific intended behavior.                                      | None.                                                                                                                                                                                                                                                                                                                                                                                                                   | AutoModel `1814c6c93a66b9d59d254960ef6a99a64249b671` can replace Nemotron V3's Mamba output projection with `LinearLoRA`, but the normal fused training branch passes `self.out_proj.weight` directly to `mamba_split_conv1d_scan_combined` and never calls the wrapper, making the adapter a silent no-op in both SFT and GRPO policy training. The architecture should either make the fused operation LoRA-aware, select its module-forward branch when the projection is wrapped, or reject this target during PEFT construction; the examples exclude it rather than patching the model.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **U-32** vLLM Qwen3.6 large synchronous generation batch    | General model/backend scaling bug.                                     | Public `additional_config.gdn_prefill_backend=triton`; repeated-update qualification passed.                                                                                                                                                                                                                                                                                                                            | NeMo RL `71a08314bc3257a15ef04b507309649e3d3f0f68` with vLLM 0.25.1 on one GB300 reproduces a sharp Qwen3.6-35B-A3B hybrid-attention failure: a synchronous 241-row raw-text request with a 64-token allowance remains at 100% GPU without reaching vLLM's request-progress logger, including when `max_num_seqs=8`, while an actually submitted 8-row request completes in 7.95 seconds and the same model/backend completes 61 small batches at about 1.3k generated tokens/s. Automatic backend selection resolves to FlashInfer TRTLLM unquantized MoE, FlashInfer GDN prefill, and TRTLLM-generation decode in both cases. The generation stack must make large submitted batches progress with bounded scheduler concurrency, or fail diagnostically, without requiring callers to split the request before `LLM.generate`; the public Triton/FLA GDN prefill selector subsequently passed repeated GRPO updates with concurrency 64 and all 260 validation rows, without private framework code (follow-up below). |
| **U-33** NeMo RL fp32-master-weight policy load             | General policy-memory behavior, not a modality seam.                   | NeMo RL's AutoModel policy setup forces `torch_dtype=torch.float32` at model init ("Always load in float32 for master weights", `nemo_rl/models/automodel/setup.py`); measured on the example's 8×H100 host (2026-09-03), the one-rank Lightning 30B policy OOMs at ~77 GiB during init on 80 GB cards where a BF16-native load peaks at 58.8 GiB, and `dtensor_cfg.cpu_offload` fails closed for single-GPU AutoModel. | Lightning GRPO stays on ≥141 GB HBM parts (the GB300-qualified path); adopt a bf16-load opt-out for frozen-base LoRA policies if NeMo RL adds one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **U-34** Empty-reasoning chat token boundary                | General SFT tokenization/masking contract; temporary local seam.       | `FeatureCollator._encode` preserves the generation prefix for suffix supervision when tokenization merges across the prompt boundary.                                                                                                                                                                                                                                                                                   | Transformers 5.12.1's native Qwen3.6 rendering is text-prefix-stable but not token-prefix-stable (newline token 198 becomes double-newline token 271). Stitch's old exact-token-prefix assumption rejected it; this is not a tokenizer defect. The fallback encodes the rendered continuation separately and requires exact decoded-text preservation. Intended upstream home: Transformers chat tokenization, or a public AutoModel SFT continuation utility. Delete the fallback when the supported framework pin exposes equivalent prompt-preserving encoding/masking and passes the parity, EOS, and feature-index regressions below. No demo-local collator or template rewrite.                                                                                                                                                                                                                                                                                                                                    |
| **U-38** NeMo RL explicit vLLM executor selection           | General generation-topology control.                                   | NeMo RL `4d969c93268fda1687fed8ca38668da4087ce76b` still overwrites `distributed_executor_backend` after applying public `vllm_kwargs`, selecting Ray whenever TP or PP exceeds one.                                                                                                                                                                                                                                    | Keep the narrow encoder-worker override for artifacts qualified with vLLM multiprocessing. Delete it if NeMo RL exposes an explicit validated executor selection; pursue upstream only with evidence that the topology matters beyond one application.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **U-39** Interleaved projected-item transport               | Resolved local packing/formatting bug; formatter superseded by U-4.    | None.                                                                                                                                                                                                                                                                                                                                                                                                                   | NeMo RL `71a08314` preserved ordered `PackedTensor` segments, proving the original rank-three encoding was unnecessary. With U-4 adopted, producers instead place vLLM's ordered item list directly in `vllm_multi_modal_data`; the local segment formatter is deleted. Keep the frozen KERMT comparison checkout unchanged because its hash is a result input.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **U-40** Projected tensor-view serialization                | Historical packed-rollout workaround; superseded by U-4 adoption.      | None.                                                                                                                                                                                                                                                                                                                                                                                                                   | The old NeMo RL `a0a4b901` / Ray route serialized each tensor view with its entire backing storage. Local commit `b637021` compacted projected items before packing. That helper was dropped when rebasing onto native `vllm_multi_modal_data`; producers now supply ordered items directly. The historical payload-size result does not qualify serialization on the new transport.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **U-41** AutoModel local-vocabulary training logprob memory | Local H100 qualification passed; upstream candidate not merged.        | NeMo RL `a0a4b901` does not forward `policy.logprob_chunk_size` from AutoModel LossPostProcessor; its unsharded loss materializes full FP32 log_softmax. A 16k FSDP/EP8 H100 GRPO batch failed allocating another 6.94 GiB with 1.01 GiB free.                                                                                                                                                                          | Isolated RL commit `aac5bdbdd07336e890ff7291eb6ec1df1f13da1b` wires the existing chunked vocabulary loss through the singleton TP group. No Stitch/demo loss patch. Validate probabilities, gradients, and full-batch updates before adopting the image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **U-42** NeMo RL positive-example NLL reward transport      | General training-loss wiring gap; not Stitch modality scope.           | Pinned synchronous GRPO constructs `ClippedPGLossDataDict` without `rewards`, while positive-example NLL requires that field and otherwise silently stays zero.                                                                                                                                                                                                                                                         | NeMo RL should carry a validated per-response reward/correctness signal into policy loss batches or reject the enabled option. No demo workaround; see the pinned reproduction below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **U-44** vLLM Mamba2 batch-invariant evaluation             | Open; general hybrid-model backend support, not Stitch modality scope. | vLLM 0.25.1 rejects `VLLM_BATCH_INVARIANT=1` for `MAMBA2_ATTN`. The same fixed Lightning checkpoint scored 240 then 238 on greedy development evaluation with identical inference configs.                                                                                                                                                                                                                              | Upstream vLLM should qualify batch-invariant Mamba2 scan/convolution/attention execution and expose support through its existing backend capability. Stitch already passes the public setting; no demo guard bypass or kernel shim. See reproduction below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

`MultimodalInputMixin.forward`, `ProjectorSidecarState`, portable projector
artifacts, feature-cache formats, the replicated-projector protocol, and the
vLLM model/processor overrides classified above are package or model contracts,
not missing framework hooks.

Frozen projector replicas now have short-run AutoModel policy qualification
with eight FSDP/EP ranks and TP=CP=1, including compact-LoRA warm starts,
nonzero updates, and checkpoint reloads; see the dated qualification below.
vLLM generation supports TP model sharding and colocated `EP=TP`. Unsupported
pipeline/context-parallel policy payloads, vLLM pipeline parallelism and
`EP!=TP`, mixed native and external media in one batch, and optimizer resume
for jointly trainable LoRA plus a projector fail closed. Large-context capacity
is a separate application qualification gate.

### U-32 follow-up: sustained Qwen GRPO, 2026-09-09

The KERMT two-model comparison reproduced a second-rollout stall after a
successful optimizer update, including with eager vLLM execution. Pins: NeMo RL
`71a08314bc3257a15ef04b507309649e3d3f0f68`, vLLM 0.25.1, AutoModel
`1814c6c93a66b9d59d254960ef6a99a64249b671`, Transformers 5.12.1, Torch
2.11.0+cu130, Stitch `a9d65b1730fccb155550a7aa08db856716471d24`.
The GB300 demo image is
`sha256:ddbf9321519c904e45c177ce0913d20a904ff2b70d9f8c34fcd571b02a1fc90b`.

Smallest observed failing sequence: initialize the 72-update Qwen SFT adapter
and frozen projector, generate 16 prompts × 16 responses at total context 2048,
apply one GRPO update, refit, then generate the next 256-response batch.
`enforce_eager=true`, `max_num_seqs=64`, and dynamic training envelope 8192
complete update 1 (464.41 s, zero truncations, mean 262.39 generated tokens),
but update 2 produces no completed requests for over six minutes at 100% GPU.
Two stack snapshots show the same tensors and 5,872-token / 64-request prefill,
blocked in native `fused_add_rms_norm` under Qwen3.5's vLLM decoder; the engine's
completed-output list remains empty. This is an observed stack location, not
an isolated kernel root cause. A prior two-process optimizer/resume/validation
smoke passed; it did not qualify sustained generate/train/refit cycles.

Reproduce from `../examples/nemotron-kermt` using its documented
GB300 container mounts and
`python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-qwen3.6-35b-a3b.yaml logger.wandb_enabled=false`.
Exact launch config and input hashes are in
`outputs/grpo-qwen3.6-r64-p16-g16-t2048/launch-{config.yaml,provenance.json}`;
the log is `outputs/grpo-qwen3.6-r64-p16-g16-t2048-20260909.log` and stack dumps
are `outputs/grpo-qwen-full-step2-stall-worker[-second]-20260909.txt`.
The attempt was deliberately stopped before its first periodic checkpoint.

Needed behavior: repeated synchronous external-embedding generation must make
progress after policy refit with a bounded scheduler batch, or fail with a
diagnostic. The existing public backend selector below meets the application's
need on this pin; a new Stitch caller-batching hook is not required.
The demo must not copy `LLM.generate`, replace refit/worker lifecycle methods,
or split private request lists locally. No such workaround was implemented;
the auto-selected FlashInfer failure remains recorded for upstream diagnosis.

The same pinned vLLM also exposes the public
`additional_config.gdn_prefill_backend="triton"` selector. The Qwen application
recipe now sets it under `policy.generation.vllm_kwargs.additional_config`;
no Stitch implementation change is needed to select the backend. The short
repeated-update qualification passed; the failure above used `auto`,
which selected FlashInfer on GB300. Such a probe needs no local framework
workaround and may qualify an existing path without changing the package pin.

The unique `grpo-qwen-triton-prefill-qualification-20260909-1736` probe
confirmed that the selector reaches Triton/FLA, saved update 1, and completed
the second rollout that stalled with FlashInfer. A stack snapshot recorded
completed requests and ordinary 64-token decode, replacing the wedged
5,872-token prefill. Thus the reproduced U-32 failure has an application-config
solution through the existing public API. The subsequent reference-policy
log-softmax OOM (30.19 GiB requested, 26.33 GiB free) is a separate capacity
issue; Qwen now uses a 16,384-token reference-logprob microbatch envelope,
without changing its logical GRPO batch. Resume completed updates 2 and 3,
all 260 validation rows, and checkpoint 3 with exit 0 on 2026-09-09. Across
the three successful training rollouts, none of 768 responses was truncated.
The unique completion log is
`nemotron-kermt/outputs/grpo-qwen-triton-prefill-qualification-20260909-1736-resume16k.log`
in the demos repository. Full 198-update training was qualified separately below.

A subsequent compiled resume (`enforce_eager=false`, same Triton selector and
microbatch budgets) also completed updates 4–5, all 260 validation rows, and
checkpoint 5 with exit 0. Validation took 28.59 seconds. Both additional
256-response rollouts had zero reported truncations. The Qwen recipe therefore
keeps normal compiled generation; eager mode is not required for this solution.
Evidence: `outputs/grpo-qwen-triton-prefill-qualification-20260909-1736-compiled.log`
and `compiled-qualification-result.json` under the same qualification root in
the demos repository.

The full compiled Qwen run completed 198 optimizer updates on 2026-09-10 with
exit 0, retaining the public Triton selector, concurrency 64, and 16,384-token
reference-logprob microbatches. Its 50,688 retained responses contained 38
truncations; final validation processed all 260 rows in 31.66 seconds. An
unrelated host-memory threshold failure during the first attempt required
resuming checkpoint 125; the unchanged resume completed through checkpoint 198.
This qualification used frozen Stitch `a9d65b1730fccb155550a7aa08db856716471d24`
and the same NeMo RL `71a08314` / vLLM 0.25.1 stack above, not a subsequent pin
bump. Input/checkpoint hashes, retained optimizer lineage, and the successful
exit are recorded in the demos repository at
`nemotron-kermt/outputs/grpo-qwen3.6-r64-p16-g16-t2048/comparison-training-result.json`.
No Stitch source or demo-local framework workaround was required for U-32.

### U-34 reproduction: Qwen empty-reasoning SFT, 2026-09-10

The KERMT answer-style experiment qualified native empty-closure supervision on
Lightning, then found this failure before starting the analogous Qwen training.
Pins: Stitch `a9d65b1730fccb155550a7aa08db856716471d24`, Transformers 5.12.1,
Qwen/Qwen3.6-35B-A3B tokenizer revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, demo image
`sha256:ddbf9321519c904e45c177ce0913d20a904ff2b70d9f8c34fcd571b02a1fc90b`.
The failure reproduces without any encoder features or GPU allocation:

```python
from transformers import AutoTokenizer
from nemotron_stitch.automodel.data import FeatureCollator

tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/models/qwen3.6-35b-a3b", local_files_only=True
)
collator = FeatureCollator(
    tokenizer,
    projector_name="feature",
    placeholder_token_id=248055,
    max_length=2048,
    chat_template_kwargs={"enable_thinking": True},
    supervision="suffix",
    truncate=False,
)
collator._encode({"prompt": "How many atoms?", "target": "<think>\n</think>((2))"})
# ValueError: chat template is not prefix-stable between prompt and full rendering
```

The native generation prompt ends `assistant\n<think>\n`; the full native
assistant message contains `assistant\n<think>\n\n</think>\n\n((2))`.
Token 13 is therefore 198 (`\n`) in the prompt but 271 (`\n\n`) in the full
sequence. Qwen's template trims the reasoning body before adding those newlines,
so tagged content or a whitespace-only reasoning field cannot remove the
boundary merge. Using reasoning-off SFT would instead mask the closing tag;
that is a different supervision contract from the qualified Lightning arm.

### U-34 temporary seam and deletion condition

`FeatureCollator._encode` now keeps the existing encoding unchanged whenever the
prompt token IDs are already an exact prefix. Only for `supervision="suffix"`,
a token-prefix mismatch may use a continuation encoding:

1. Require the full native rendering to start with the exact prompt text.
2. Keep the prompt token IDs and tokenize only the remaining text with
   `add_special_tokens=False`.
3. Require both the prompt and the concatenated IDs to decode exactly to their
   native rendered texts, with special tokens retained and cleanup disabled.
4. Mask the original prompt IDs and supervise every continuation token, including
   the native closing tag and EOS. Apply the existing length cap afterward.

This deliberately permits a non-canonical tokenization of the full text: it is
an autoregressive continuation of the actual inference prefix, not a re-encoded
prompt. It inserts no extra text or special tokens and makes no template edits;
its token count can differ from canonical full-text encoding. Non-prefix text,
lossy continuation encoding, and token-prefix mismatches under `assistant_content`
still fail closed. The latter mode is not extended to multi-turn continuation encoding.
Feature positions continue to be derived from the resulting input IDs; the
forward kwargs, index geometry, and artifact schema are unchanged.

This is a temporary exception inside the existing feature collator, not a new
training framework or a claim that general SFT masking is modality work. The
preferred upstream owner is Transformers' chat tokenization; AutoModel's public
SFT formatting utilities are another plausible owner. Full-render assistant
masks alone do not meet the requirement because they do not preserve the
inference prompt IDs. No upstream PR is linked yet.

**Delete, rather than retain a compatibility branch**, when a supported pinned
Transformers/AutoModel public API implements this continuation contract. Adopt
that API in the collator and gate removal on unchanged IDs/labels for existing
prefix-stable rows, native empty-closure and EOS supervision, exact prompt/text
preservation, ragged/padded/packed feature-index alignment, and rejection of
incompatible templates. Recheck the public APIs on pin bumps. Applications
should adopt the qualified Stitch revision; they must not replace `_encode`,
rewrite the chat template, or patch their frozen comparison checkout. This seam
unblocks data encoding, not by itself a new SFT/GRPO training qualification.

Local qualification (2026-09-10): `tests/test_chat_continuation.py` covers the
boundary merge, exact stable-row parity, loss masks/EOS, lossy and incompatible
render rejection, length caps, and ragged/padded/packed feature alignment. Its
optional native-tokenizer test ran against KERMT's local Qwen snapshot with
Transformers 5.12.1; the original empty-reasoning reproduction now passes,
including native prompt token 198, supervised `\n</think>\n\n((2))`, and EOS.
It also verifies unchanged IDs/labels for native prefix-stable reasoning-on
and reasoning-off examples. No model weights or GPU are needed for these tests.
Set `STITCH_TEST_QWEN_TOKENIZER` to the local snapshot path inside the test
container to enable this offline test (otherwise only that test skips).
Tokenizer/template SHA-256 values used:

- `tokenizer.json`: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- `tokenizer_config.json`: `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b`
- `chat_template.jinja`: `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`

Full `tests/` runs in the prescribed `main-a0a4b901-clean` base image, with the
native test enabled: driver **322 passed / 14 skipped**, AutoModel policy venv
**327 passed / 9 skipped**, generation venv **328 passed / 8 skipped**. Skips
are unavailable frameworks per venv and CUDA-only tests. Ruff and diff checks
pass; pyrefly reports the same four errors as an isolated unchanged `HEAD`
check. KERMT pin adoption, full-dataset preparation, and GPU training remain
separate steps; neither the demo nor its frozen comparison checkout was edited.

Exact rendered texts and token IDs are retained in the demos repository at
`nemotron-kermt/outputs/lightning-answer-style-20260910/qwen-empty-trace-boundary/reproduction.json`.
Lightning SFT and its independent GRPO follow-up remain unblocked.

### U-4 follow-up: generation-only eval drops encoder payloads, 2026-09-11

Pins: NeMo RL `a0a4b901c5b66b70cbf16c2d45c437d041706aba`, vLLM 0.25.1,
and Stitch `4c7279b53acc234c2d0086f125bf76475c3dc2f5`. A processed
encoder datum contains its projected tensor as a one-row `PackedTensor` under
`vllm_encoder__rna` in the message log. Passing that datum through NeMo RL's
public generation-only `setup` / `run_env_eval` path removes the field in
`eval_collate_fn`. `_run_env_eval_impl` then creates a prompt containing
only `{"prompt": vllm_content}`, because its multimodal include-list names
only image, audio, and video. The model consequently runs without its encoder
conditioning and raises no error.

BioReason-RNA provides the observable reproduction: the original checkpoint's
private vLLM evaluator scored 161/350 (46.0%) on the first internal-predictive
rows, while the NeMo RL path scored 111/350 (31.7%). Direct vLLM generation
with the converted checkpoint, identical prompt, and identical projected
tensor matched the private model's first 64 token IDs. Inspection of the NeMo
RL batch and worker request then established that its divergent trajectory had
received no tensor. The demo would otherwise need to patch NeMo RL's collator
and private eval loop or copy its entire environment runner.

Stitch temporarily exposed `encoder_eval_collate_fn` to extract each
adapter's logical projected row and build vLLM-native prompt objects. RL#3803
merged as `4d969c93` on 2026-09-16 and now preserves
`vllm_multi_modal_data` through collation, rollout, formatting, and
generation-only evaluation. The `4d969c93` pin adopts that field directly;
the temporary collator and prompt-formatter seam have been deleted.

Local qualification: the corrected converted checkpoint scored 9,151/20,176
(45.3559%) exact 5-way on the complete internal-predictive split, with 99.9752%
parse rate. The private project's reported one-generation result is 45.67%; the
0.31-point difference is below one binomial standard error (0.35 points). The
run used `TP=EP=8`, FP16 Mamba state, 1,024/256 outer/active batching, and
decode-only CUDA graphs on eight H100 80GB GPUs. The result and strict identity
artifacts are in the demos repository under
`outputs/reference-rlmerged-r8kl001/evaluation/payload-fixed-full-20176`.

### U-39 resolution: explicitly interleaved projected positions, 2026-09-11

Pins: Stitch `a9d65b1730fccb155550a7aa08db856716471d24`, NeMo RL
`71a08314bc3257a15ef04b507309649e3d3f0f68`, vLLM 0.25.1, and demo image
`sha256:ddbf9321519c904e45c177ce0913d20a904ff2b70d9f8c34fcd571b02a1fc90b`.
The first experimental implementation encoded item boundaries as a rank-three
`[spans, tokens_per_span, hidden]` tensor, then split its leading axis in the
Stitch prompt formatter. An audit of the cited NeMo RL revision found that its
public `PackedTensor` API already carries multiple ordered segments in one
logical row through collation, slicing, and GRPO batching. vLLM already accepts
those segments as an ordered item list. The missing boundary was therefore a
local producer/formatter contract bug, not an upstream gap.

The initial correction consumed `iter_logical_segments()` directly and kept
the fallback pin's materialized tensor as one item. U-4's later adoption made
that formatter obsolete: producers can now place each non-empty rank-two
`[tokens, hidden]` item in an ordered list under `vllm_multi_modal_data`, and
tensor rank still does not encode transport geometry.

Before that diagnosis, a two-molecule pinned-container qualification showed
that the rank-three workaround could complete one AutoModel optimizer update,
export all three projector branches, and generate through vLLM with eight
interleaved items for ethanol and two for bondless methane. That run validates
the item ordering and model path, but is not evidence of a missing framework
hook. Its first generation setup failed before generation because the configured
per-prompt modality limit was one.

The matched quality arm reached 229/260 (88.08%) on its development split. The
frozen comparison checkout remains unchanged because its exact worker hash is
an immutable input to the training and development receipts. Future runs should
use the corrected segment producer and maintained Stitch revision. The demo
continues to own prompt grammar, graph indexing, and projector selection.

### U-45: AutoModel equal-pack loss normalization (2026-09-16)

Adjacent general training capability; implementation belongs in AutoModel.
Affected pin: `1814c6c93a66b9d59d254960ef6a99a64249b671`, as installed in
`nemotron-gene:grpo` image
`sha256:de69303a81366605f2c3eec76086a309d8ddfc8081ed1b42630e85d7c32cbe91`.

The genome-research alignment reference averages labeled-token loss within
each pack, then averages packs across accumulation and data-parallel ranks.
The pinned trainer supplies a global labeled-token denominator to its loss;
public MaskedCrossEntropy and FusedLinearCrossEntropy reject mean reduction
when that denominator is supplied. Minimal reproduction, verified in the
pinned image's AutoModel policy interpreter:

```python
import torch
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

MaskedCrossEntropy(reduction="mean")(
    logits=torch.zeros(1, 2, 3),
    labels=torch.tensor([[0, 1]]),
    num_label_tokens=2,
)
# AssertionError: num_label_tokens is only supported when reduction is 'sum'
```

Needed: an explicit trainer normalization policy supporting equal-pack means
with correct gradient accumulation and distributed scaling. Stitch should
forward the upstream configuration without owning a second training loop.
The otherwise-required demo workaround would copy the reference's custom loss
and private optimizer-step override to ignore the token denominator and inject
accumulation scaling. No such workaround was added. Exact alignment replay is
paused until a pinned public implementation is available and tested against
unequal-length packs (including accumulated versus unaccumulated gradients).

### Frozen sidecar policy mesh qualification, 2026-09-12

Stitch `9306573e59c633b6ae4ff71ce906361c0d5be8fa` rejects
`policy.dtensor_cfg.expert_parallel_size: 8` before worker construction, and
separately rejects any policy world size other than one. NeMo RL
`a0a4b901c5b66b70cbf16c2d45c437d041706aba` already owns FSDP/EP policy
placement, compact-LoRA loading, and colocated rollout offload. The required
modality seam is one frozen, artifact-validated projector replica per policy
rank, retaining complete hidden vectors (TP=1, CP=1). A demo workaround would
have bypassed these guards; no such workaround is installed.

Commit `afab678061b6591f6a96aa8765c56c045650f701` relaxes only the sidecar
mesh guard, leaving distribution and optimizer behavior upstream. The KERMT
indexed and vectors-only applications each passed two eight-rank FSDP/EP8
GRPO updates with four colocated TP2/EP2 vLLM replicas. The runs verify donor
load, rank-local feature transport, refit, nonzero reward-derived advantages,
changed LoRA weights, checkpoint export, and separate-process development
evaluation after reloading those weights. All 186 adapter tensors changed in
each arm. Activation checkpointing remained off. NeMo RL's default FP32
master storage with BF16 computation is required; the older one-GPU BF16
storage override failed FSDP's mixed-dtype check and was removed from config.

The pinned image is
`nvcr.io/1015084509601350/nemotron-kermt@sha256:fe192829fad92ac501f64baca60fd90315457b8a70438d045d6bf8f50e33bc0b`.
Receipts are in the sibling demo's
`nemotron-kermt/outputs/three-scheme-h100-qualification/grpo-smoke-audit.json`
and per-arm `outputs/three-scheme-grpo-ep8-v2/` directories. These qualify the
integration seam, not model quality or the larger campaign batch's capacity.

### U-41 capacity follow-up, 2026-09-12

Chunking passed six GPU forward/gradient parity tests and allowed forward loss
at 16k context. A 1,024-token chunk then exhausted backward temporary memory;
256-token chunks reached dense-logit gradient allocation but failed requesting
6.88 GiB with 6.86 GiB free. AutoModel setup in the pinned RL revision hardcodes
FSDP output_dtype=float32, retaining full FP32 residuals/logits even with BF16
compute. Isolated RL commit `720f6701925a14cae030b9a2e9c28ecd4c393c4c` exposes
`dtensor_cfg.fsdp_output_dtype`, defaulting to float32 for compatibility. The
next probe selects bfloat16 outputs, retains FP32 masters/reductions, and keeps
activation checkpointing off. This is a precision choice, not a claimed
bit-exact model-forward refactor. Full-batch qualification remains pending.

U-41 runtime follow-up: RL `a20dc905f2dcfd41c362c5df2d0c7a5abe91a90f`
corrects the output-precision scope and adds three setup tests, all passing in
the policy interpreter. Indexed KERMT passed a 192-completion, 16k-context
FSDP8/EP8 update with BF16 outputs and no activation checkpointing. KERMT-only
also passed after selecting fixed one-sequence microbatches and 64-token loss
chunks; dynamic multi-sequence batches exceeded H100 backward memory. Both
updated all 186 LoRA tensors with nonzero supervised advantages. The default
allocator is retained: expandable allocations require `pidfd_getfd` for CUDA
IPC refit, denied by this local container's seccomp profile. No local privilege
change, demo loss patch, or refit workaround is required for the fixed-batch
path. Checkpoint evaluation and the final text-only stress gate remain in
progress in the demo's qualification record.

Final U-41 qualification: all three KERMT comparison arms passed full
192-completion GRPO updates and independent checkpoint evaluations. The three
matched eight-H100 batch campaigns were subsequently verified Running with
actual training updates on 2026-09-12. Exact pins and per-arm evidence are in
`../examples/nemotron-kermt/outputs/three-scheme-batch-20260912-v1/qualification.json`.

### U-42 reproduction: positive-example NLL on synchronous GRPO, 2026-09-13

- **Affected pin:** NeMo RL `a0a4b901c5b66b70cbf16c2d45c437d041706aba`,
  qualified KERMT image
  `nvcr.io/1015084509601350/nemotron-kermt@sha256:38927f2f67b2f9f855369e93844436f96ccc8da46fd732c6f3605677a130fbd8`.
  Inspection ran inside this image, not the host checkout. Source SHA256:
  `grpo.py=bd84f9f04519836bb2c504775526d442e366232ae773eb2a0ed144c8e93f9eb8`,
  `loss/loss_functions.py=551dff8060e8e64e28aacfc287174a3bc6cadf59f1b9176e654d370f68af4a73`.
- **Smallest missing contract:** synchronous `grpo_train` constructs `train_data`
  at line 3402 with input IDs/lengths, generation logprobs, token mask and sample
  mask. Subsequent explicit writes add previous/reference logprobs and advantages,
  but no rewards. The loss gates NLL on
  `self.positive_example_nll_weight > 0 and "rewards" in data`; its initialized
  zero survives when the field is absent. Thus enabling the config alone cannot
  exercise this loss on the inspected path. This is pinned source/AST evidence,
  not a claim that a positive-NLL training experiment was run.
- **Needed behavior:** a public NeMo RL batch contract should transport aligned
  per-response rewards or an explicit correctness mask to policy minibatches,
  including dynamic sampling and masking. An enabled but unavailable auxiliary
  objective should fail closed. No Stitch-owned modality-specific seam is needed.
- **Workaround deliberately not implemented:** adding `rewards` into native
  training-batch construction or wrapping the policy loss in the molecular demo.
  Resume this experiment only after a pinned upstream solution or supported
  public configuration passes an end-to-end nonzero-loss/gradient check.
- **Additional semantic gate:** the native mask uses `rewards > 0`; a task that
  gives positive formatting credit to wrong answers must first map rewards so
  positivity actually means correctness. Otherwise NLL would reinforce wrong
  formatted answers even after transport is fixed.

The consumer evidence record is
`../examples/nemotron-kermt/outputs/indexed-grpo-search-20260912/positive-nll-transport-audit.json`.
The consumer continues independent development evaluation and configuration-only
KL experiments; no framework or Stitch implementation was changed.

### U-43 reproduction: full decoder SFT, 2026-09-13

- **Affected pin:** Stitch `b637021216b5857392ce60f944aa357f7679d350`,
  installed as 0.4.0 in qualified image
  `nvcr.io/1015084509601350/nemotron-kermt@sha256:38927f2f67b2f9f855369e93844436f96ccc8da46fd732c6f3605677a130fbd8`.
  Installed `projector/trainability.py` SHA256:
  `044c0d88d493639977851b4a4d1caca52a6c14e1ada8aad66d2b22ca7990c9bb`.
- **Smallest reproduction:** a CPU torch module containing `decoder=Linear(2,2)`
  and `mm_projector=Linear(2,2)`, passed to
  `configure_trainable_parameters(model, 2, policy=TrainabilityPolicy(projector_patterns=("mm_projector",)), train_projector=True)`.
  Inside the pinned container this freezes both decoder parameters and raises
  `ValueError: Stage 2 has no LoRA parameters`; projector parameters remain trainable.
  Removing the application's PEFT configuration alone cannot enable full SFT.
- **Needed behavior:** a public full-decoder training selection, orthogonal to
  projector trainability and preserving existing alignment/LoRA behavior. Prefer
  AutoModel's native freeze configuration (U-7) if available at the qualified pin.
  Validate trainable families and full-checkpoint export/reload, including the
  projector sidecar handoff to RL and generation. This is modality-neutral.
- **Not implemented:** a demo optimizer override, parameter reclassification as
  extra/LoRA, or post-construction unfreeze shim. Full SFT qualification pauses
  until supported through a pinned Stitch/upstream implementation. Native RL
  exposes LoRA-disabled configuration; its full-weight encoder handoff and
  eight-H100 memory/throughput still require independent qualification.

Consumer receipt:
`../examples/nemotron-kermt/outputs/indexed-grpo-search-20260912/full-sft-trainability-audit.json`.
No full-model training or performance claim follows from this CPU reproduction.

### U-43 implementation qualification, 2026-09-13

The installed AutoModel worker resolves to
`/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel`, revision
`1814c6c93a66b9d59d254960ef6a99a64249b671`. Its native `build_model` lacks
`cfg_freeze`; the newer host checkout's API is not in this image.

Stitch now has an explicit `projector.train_decoder` stage-2 option and
`projector.trainability_decoder_patterns` name substrings. It preserves native
requires-grad for those decoder parameters, applies existing projector/extra
choices, freezes unclassified parameters, and rejects PEFT/full ambiguity,
missing matches, and full decoder alignment. Default LoRA behavior is unchanged.
This is shared package support, not a consumer optimizer workaround; replace
with native freeze selectors after U-7 adoption. No forward/artifact geometry
changed; trainability audit metadata gains `train_decoder`.

Focused trainability/recipe tests: 61 passed in each of the pinned driver and
AutoModel-policy-worker interpreters, including actual optimizer updates and
frozen encoder/protected-parameter checks. Full-model GPU capacity, complete
checkpoint export/reload, and SFT-to-RL/generation handoff remain unqualified.

U-43 full-model follow-up: commits `aba1538` and `7f82306` implement the
explicit decoder mode and require every decoder selector to match. The first
consumer smoke was rejected because HF `backbone.*` differs from native
AutoModel `model.*`; the missing-selector regression is now covered. All 62
focused tests pass in the pinned policy-worker interpreter. Native meta-model
geometry records 378 decoder tensors/31,577,937,344 parameters. Distributed
trainability matches these names exactly after removing checkpoint wrappers.

Eight-H100 full SFT at 16k packs/global batch 8, BF16 compute/FP32 masters, failed
without activation checkpointing (CUDA OOM at 78.94 GiB). Native activation
checkpointing passed 2 updates and full checkpoint export, reporting 64.22 GiB.
Streamed export audit found 6243 finite tensors and 6172 changed from the original
base, including decoder layers. Image:
`sha256:ff830dd1ef70e1e49c2304e2b0992e4465b536ed08dbb3252564eb590f49adfd`.
This qualifies short SFT execution/export, not quality, sustained throughput,
full RL, or independent reload. The consumer is now qualifying native full-GRPO
loading and generation handoff. No application/framework workaround was added.

### U-44 reproduction: Mamba2 batch-invariant evaluation, 2026-09-14

Affected stack: vLLM **0.25.1** in immutable demo image
`nvcr.io/1015084509601350/nemotron-kermt@sha256:a4c99b07d8fc950a51dd16ea83f946c84ef5b4331015a8e4d9d702cfd541681c`.
The image reports `/opt/nemo-rl` Git HEAD
`a0a4b901c5b66b70cbf16c2d45c437d041706aba`; the image digest binds all
application/framework overlays, independently of that repository metadata.

Smallest reproduced failure, in that image's vLLM worker interpreter (no weights
or GPU allocation required):

```bash
VLLM_BATCH_INVARIANT=1 /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python -c \
 'from vllm.v1.attention.selector import get_mamba_attn_backend; from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum; get_mamba_attn_backend(MambaAttentionBackendEnum.MAMBA2)'
```

Observed: `RuntimeError: VLLM batch_invariant mode is not supported for MAMBA2_ATTN.`
The public selector in `vllm/v1/attention/selector.py` rejects the backend's
`supports_batch_invariance()` capability. The configured Triton MoE expert
backend supports the mode, but that does not establish hybrid Mamba2 support.
The same error occurred during a real TP2/EP2 Lightning generation startup on
8xH100; job `kermt-ring-invariant-donor-0914-7bv6`. The duplicate candidate job
was stopped before execution once this architecture-level failure was known.

Needed behavior: reproducible greedy evaluation for the supported hybrid model,
including recurrent state/scan and convolution kernels as well as attention,
MoE and collectives. This belongs in vLLM's existing batch-invariant backend
contract. Stitch needs no new modality-specific abstraction or owned kernel.
The existing public setting already reaches its generation worker correctly.

A temporary demo workaround would have replaced the backend capability predicate
or kernel implementations. Neither is implemented: claiming support by bypassing
the guard would not prove numerical invariance. Resume strict batch-invariant
qualification after a pinned upstream implementation is available. Ordinary
supported evaluation can continue, with repeated outputs and score variability
reported explicitly.

Consumer artifacts: `../examples/nemotron-kermt/outputs/ring-evidence-invariant-eval-v1/`
and `experiments/ring-evidence-v1/REGRESSION_DEBUG.md`. The initial candidate's
240/260 and fresh 238/260 are both retained; no deterministic or robust-improvement
claim follows from the better score alone.

### Rebase validation, 2026-09-16

The retained FSDP/EP policy and full-decoder SFT changes pass the full CPU suite
against NeMo RL source `4d969c93268fda1687fed8ca38668da4087ce76b`: driver
329 passed / 15 skipped, AutoModel worker 334 passed / 10 skipped, and vLLM
worker 335 passed / 9 skipped. Ruff 0.16.4 passes and Pyrefly 1.2.0 reports
zero errors. GPU kernels and the opt-in tokenizer fixture were not exercised.

At rebase time, both documented `main-4d969c93-clean` image tags returned
manifest-not-found. At the BioNeMo import, the arm64 tag resolves but the amd64
tag remains unavailable. These checks used
`main-71a08314-clean-cuda13.1-amd64` with the exact new NeMo RL source mounted
ahead of installed packages on `PYTHONPATH`, not a qualified replacement image.
The unmodified older driver image passed 328 tests and failed only the new
native multimodal transport regression; the source overlay passes that
regression. Publish and qualify the amd64 base before this recipe's container CI
can pass.
