# Minimal LLaVA example

A small, reproducible vertical slice that adds a frozen CLIP encoder and a
LLaVA-style projector to several base models, then runs all three
post-training stages end to end. The qualified control is text-only
Nemotron-3-Nano-4B. The Qwen paths demonstrate the same seams with non-NVIDIA
bases: dense Qwen3.5-4B and Qwen3.6-35B-A3B using AutoModel's native MoE
backend. Both keep Qwen's built-in vision tower frozen and unused.

1. **alignment** — train only the image projector (`mlp2x_gelu`, 768 → the
   base model's hidden width) against the frozen language model;
2. **SFT** — warm-start the projector from the alignment artifact and train
   it jointly with a LoRA adapter on the language model's attention and
   mamba-input projections, plus the shared expert on the MoE target — every
   projection family that survives the GRPO handoff; the SFT configs document
   the excluded ones (the LLaVA finetune stage); and
3. **GRPO** — warm-start SFT's projector and LoRA adapter and keep updating
   only the adapter, so the whole recipe carries a single adapter, never a
   merged copy.

Serving is part of the design. GRPO's rollout engine is plain vLLM: at every
update the current adapter is merged into the base weights and refit into the
running engine (CUDA-IPC colocated, NCCL across two GPUs), and each prompt's
image reaches vLLM as 49 already-projected soft tokens — the projector
(frozen again at this stage) runs in the rollout data path over cached CLIP
features. The CLIP encoder runs once per training row during preparation, not
inside training or GRPO rollouts. vLLM never imports CLIP, sees a raw image, or
holds a separate adapter object. Inference on a new image runs the same frozen
encoder once before calling vLLM.

Everything modality-neutral — preparation, cache, dataset, collator, projector,
scatter, artifact codec, training recipe, RL transport, and workers — is in the
[`nemotron-stitch`](../../README.md) package. The common application code is
[`llava_example.py`](llava_example.py): encoder and dataset access, row
normalization, and the Nemotron bindings. [`llava_qwen.py`](llava_qwen.py) is
the small Qwen-specific model-registration boundary. Configs hold the
values their frameworks consume directly — model/projector construction,
token IDs, LoRA settings, paths, and resources. Shapes, per-row token counts,
and artifact provenance are not duplicated into YAML. Scripts run or qualify
the example but contain no application contract to copy.

**Adding your own modality?** Read [`quickstart.md`](quickstart.md) first: it
walks the three stages with the exact pieces you would change, written for
domain scientists rather than framework experts.

| Role                        | Selection                                                                                                                            | Pinned revision                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| Language model (4B control) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`                                                                                              | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` |
| Language model (non-NVIDIA) | `Qwen/Qwen3.5-4B`                                                                                                                    | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Language model (MoE smoke)  | `Qwen/Qwen3.6-35B-A3B`                                                                                                               | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Language model (target)     | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`                                                                                  | `b3caaabed0263651a17dc1f2d4ce97e794f76c44` |
| Frozen encoder              | `openai/clip-vit-base-patch32`                                                                                                       | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Data                        | `lmms-lab/LLaVA-OneVision-Data`, configurations `diagram_image_to_text(cauldron)` (alignment) and `CLEVR-Math(MathV360K)` (SFT/GRPO) | `7ca5e5bf8b2006d5dfa0549756198474f0897f63` |

## Data terms

The OneVision repository is labeled Apache-2.0 on its dataset card **and**
separately limits use to academic research and education; this example
surfaces both statements and does not try to reconcile them. The selected
components are `diagram_image_to_text(cauldron)` (from the CAULDRON
collection) and `CLEVR-Math(MathV360K)` (from MathV360K). Confirm the terms
fit your use before running preparation. No source image or dataset row is
committed to this repository.

## Setup

The environment (unmodified NeMo RL main, nesting AutoModel r0.6.0, vLLM
`0.25.1`, torch `2.11.0+cu130`, transformers `5.12.1`) is built as a
container from the repository root:

```bash
docker build -f examples/llava/Dockerfile -t llava-example .

# Any GPU SKU: --gpus all exposes every device (restrict a subset with
# --gpus '"device=0,1"'). --cap-add SYS_PTRACE is for the colocated GRPO
# stage's CUDA-IPC weight refit (pidfd_getfd). The mounts persist the HF
# cache, the model snapshots, and the training outputs on the host.
docker run -d --name llava-example --ipc=host \
  --gpus all --cap-add SYS_PTRACE \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/examples/llava/artifacts:/opt/llava-example/artifacts" \
  -v "$PWD/examples/llava/outputs:/opt/llava-example/outputs" \
  llava-example
```

The base is a public mirror of NVIDIA's NeMo RL nightly at `b03da0f4`, pinned
in the Dockerfile to the multi-platform digest
`sha256:a92eb4efb488be6814ea7236b9b813415b8e7a814e1146389c7b15ad2d682f2a`.
The public mirror is currently AMD64, matching BioNeMo CI runners. The example
uses the base's prebuilt
AutoModel worker environment for preparation, alignment/SFT, and the GRPO
controller; NeMo RL selects separate environments for its Ray workers.

On hosts whose NVIDIA driver predates CUDA 13, also mount the CUDA
forward-compat shim (`-e LD_LIBRARY_PATH=/opt/cuda-compat -v /path/to/cuda-compat:/opt/cuda-compat`; any CUDA-13 container's
`/usr/local/cuda/compat` provides it). For a quick host-side setup without
Docker, install the root package with `pip install -e .` in an environment
carrying those pins and put `examples/llava` on `PYTHONPATH`; the Dockerfile is
the qualified path.

Inside the container, fetch the immutable snapshots and build the data cache.
The language model downloads straight to the artifact path every stage's
config references (`--local-dir` writes real, resumable files); the encoder
stays in the HF cache, where `llava_example.py` resolves it by repo id and
pinned revision:

```bash
docker exec llava-example hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f \
  --local-dir /opt/llava-example/artifacts/models/nemotron-nano-4b-bf16
docker exec llava-example hf download openai/clip-vit-base-patch32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
docker exec llava-example python -m llava_example
```

Preparation streams a bounded sample (the `PARTITIONS` table in
`llava_example.py`: valid-row ordinals 0–79 per training configuration, plus
80–159 for the GRPO slice; scan capped at 512 rows), runs the frozen CLIP
encoder once, and retains only the package feature cache plus a JSONL manifest — no
image files. It writes `outputs/data/{feature-cache,manifest.jsonl,...}`,
streams through a preparation-owned temporary datasets cache, and never
touches your global one. Budgets: 512 MiB transient, 64 MiB retained
(measured: 36.5 MB retained). The Nemotron targets share `outputs/data`.
Qwen uses `outputs/data-qwen` because its tokenizer needs a different set of
single-token sentinels. Pass those values and the output path directly to the
same preparation entry point; the underlying CLIP features and row selection
are otherwise identical.

## Training

The 4B control's three stages run on one GPU with at least 80 GB HBM:

```bash
docker exec llava-example python -m nemo_automodel.cli.app configs/alignment.yaml --nproc-per-node 1
docker exec llava-example python -m nemo_automodel.cli.app configs/sft.yaml --nproc-per-node 1
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo.yaml
```

The SFT checkpointer maintains `checkpoints/LATEST`, which the GRPO config reads
directly; there is no checkpoint path to copy by hand.

The single-GPU GRPO configs use the same measured residency policy as the
KERMT recipe: policy and vLLM base weights stay in HBM after the mandatory
startup handoff, vLLM gets an explicit 4 GiB KV cache, and synchronous
one-update rollouts use `force_on_policy_ratio` instead of recomputing current
policy log-probabilities. Each rollout batch is submitted to vLLM whole. The
per-step full-policy refit is consequently CUDA-IPC-local between resident
workers rather than a repeated base-model round trip through host memory. The
two-GPU config explicitly disables these colocated overrides and uses NeMo
RL's ordinary dedicated-worker lifecycle. GRPO keeps prefill and its projected
embeddings eager, but captures full decode CUDA graphs for sustained rollout
throughput. It explicitly disables Inductor compilation, retaining vLLM's eager
kernels because Lightning's compiled-kernel path measured worse policy/rollout
parity. Graph capture is bounded to the configured rollout batch; its one-time
startup cost is expected to dominate very short smoke runs.

Alignment and SFT use token-budget packing on the native Lightning and Qwen
tracks. The shared iterable dataset scans the JSONL manifest lazily, retains a
bounded buffer of raw rows for shuffling, and tokenizes and loads cached
features only as rows are drawn into the current pack. Its checkpoint state
includes the source offset, shuffle RNG and buffer, and partial pack for exact
resume. Position IDs reset at every sample boundary, while cached CLIP
features are concatenated and their projector indices are rebased into the
packed row.

Lightning uses padding-free THD with Transformer Engine. Qwen uses AutoModel's
supported NEAT form: one indexed token row, greedily filled and then padded to
the configured budget, with a block-causal mask preventing attention across
source-sample boundaries. Dense Qwen and Lightning use 2048-token packs;
Qwen3.6 uses 512 for its one-step GB300 smoke. The Nano configs intentionally
remain unpacked because their decorated Hub implementation does not accept
packed-sequence boundaries. This is online next-fit packing over the shuffled
stream; unlike the old eager path, it does not scan the full corpus for
globally tighter bins.

> **Qwen3.5-4B.** The requested checkpoint is itself a native VLM, not a
> text-only LM. This path decorates AutoModel's native
> `Qwen3_5ForConditionalGeneration` class, loads the downloaded Hugging Face
> snapshot directly, leaves its vision tower frozen and unused, and names the
> external CLIP route `clip_image` so it cannot collide with Qwen/vLLM's
> native `image` modality. No checkpoint conversion or augmented init
> directory is involved.
>
> ```bash
> docker exec llava-example hf download Qwen/Qwen3.5-4B \
>   --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
>   --local-dir /opt/llava-example/artifacts/models/qwen3.5-4b
> docker exec llava-example python -m llava_example \
>   --output-dir outputs/data-qwen \
>   --projector-name clip_image \
>   --start-token '<|vision_start|>' \
>   --placeholder-token '<|vision_pad|>' \
>   --end-token '<|vision_end|>'
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-qwen.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-qwen.yaml --nproc-per-node 1
> docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-qwen.yaml
> ```
>
> The placeholder spec reserves the 49 token positions where the external
> CLIP projector scatters its soft tokens. Qwen already defines
> `<|vision_start|>`, `<|vision_pad|>`, and `<|vision_end|>` as single tokens;
> its native image and video processors use the distinct `<|image_pad|>` and
> `<|video_pad|>` payload tokens, so this does not masquerade as a native Qwen
> image. The same boundaries are consumed by the policy collator and vLLM
> prompt replacement.
>
> These configs exercise the upstream Qwen3.5 implementations already present
> in AutoModel and vLLM. Packed one-step GPU smoke runs of alignment and SFT
> were recorded on 2026-09-02, and a GRPO smoke was recorded on 2026-09-01;
> unlike the two Nemotron tracks, a full configured training run has not yet
> been recorded.

> **Qwen3.6-35B-A3B native MoE smoke.** This is the same three-stage contract,
> but the training host directly decorates AutoModel's native
> `Qwen3_5MoeForConditionalGeneration`; it does not use Transformers' model
> implementation or convert the checkpoint. The deliberately small 512-token,
> one-step configs are an end-to-end integration run for one 256 GiB GB300,
> not a convergence recipe. GRPO uses 4 prompts × 4 generations so the fixed
> refit is amortized across 16 rollouts. They reuse the Qwen feature cache
> created above.
>
> ```bash
> docker exec llava-example hf download Qwen/Qwen3.6-35B-A3B \
>   --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
>   --local-dir /opt/llava-example/artifacts/models/qwen3.6-35b-a3b
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-qwen3.6-35b.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-qwen3.6-35b.yaml --nproc-per-node 1
> docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-qwen3.6-35b.yaml
> ```
>
> Alignment and SFT explicitly select AutoModel's portable torch expert and
> dispatcher backends for the single-GPU run. GRPO uses the same native model;
> AutoModel falls back to its standard grouped-expert implementation because
> expert parallelism is one. The policy remains GPU-resident beside vLLM on
> the qualified 256 GiB card.

> **Lightning 30B target.** The same three stages with the `-lightning`
> configs, on one GPU with ~100 GiB HBM (measured peak ~92 GiB). Lightning
> loads the pinned Hugging Face snapshot directly. The host composes the
> native family state-dict adapter with the package's projector filter, so the
> fresh module-owned projector is initialized outside the base checkpoint:
>
> ```bash
> docker exec llava-example hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
>   --revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 \
>   --local-dir /opt/llava-example/artifacts/models/lightning-30b-bf16
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-lightning.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-lightning.yaml --nproc-per-node 1
> docker exec llava-example bash -c \
>   'CUDA_VISIBLE_DEVICES=0 \
>      python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-lightning.yaml'
> ```
>
> The Lightning config also reads its automatically maintained SFT
> `checkpoints/LATEST` link. It uses the same prepared `outputs/data` as the
> 4B configs; the `-lightning` files track the `LIGHTNING_30B` spec and keep
> Lightning-specific output directories and run recipe, with the GRPO
> per-model contract composed from `configs/models/lightning-30b.yaml`.
>
> On an 8×H100 80 GB node (x86_64), alignment and SFT run expert-parallel
> across all eight GPUs — EP=8 shards the 128 routed experts (16/rank) and
> FSDP2 shards the rest, for ~13.6/~12.8 GiB per-rank peaks (qualified
> 2026-09-03, same configs, CLI overrides only):
>
> ```bash
> docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/alignment-lightning.yaml \
>   --distributed.ep_size=8
> docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/sft-lightning.yaml \
>   --distributed.ep_size=8
> ```
>
> (No batch override needed there: the packed microbatch is 1, so the global
> batch of 16 yields 2 accumulation steps at 8 ranks.)
>
> Lightning GRPO does not fit on 80 GB cards: NeMo RL loads the policy with
> fp32 master weights and the package's sidecar policy is qualified for
> exactly one rank, so a 30B one-rank policy needs ≥141 GB HBM (measured
> OOM at ~77 GiB during init where a BF16 load peaks at 58.8 GiB; see
> [`docs/upstream-gaps.md`](../../docs/upstream-gaps.md)). Multi-rank policy (EP/TP) is the
> module-ownership path owned by consumer workers (genome-research), not the
> example.

## Distributed Training

Same files, no second implementation — two GPUs instead of one. Alignment/SFT
run FSDP2 data parallel (global batch and seed unchanged). GRPO runs a
one-rank policy cluster on one GPU and a one-rank vLLM cluster on the other.
That GRPO topology is pipeline/resource parallelism across GPUs, **not**
model sharding — each cluster stays one rank, inside the package's qualified
sidecar-worker boundary. The two-rank AutoModel runs use `torchrun` directly:
the CLI's `--nproc-per-node` wrapper maps out-of-tree recipe targets to
in-repo script paths.

```bash
docker exec llava-example torchrun --nproc-per-node 2 -m nemo_automodel.cli.app configs/alignment.yaml
docker exec llava-example torchrun --nproc-per-node 2 -m nemo_automodel.cli.app configs/sft.yaml
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-2gpu.yaml
```

Eight GPUs (qualified on one 8×H100 node): alignment/SFT stay FSDP2 data
parallel and shrink the micro-batch so the global batch stays divisible by
`local_batch_size × world_size` — a CLI override, not a config fork. GRPO
keeps the policy at its qualified one-rank boundary and scales rollout to
seven TP=1 vLLM replicas, with NCCL refit across all eight GPUs
(`configs/grpo-8gpu.yaml`):

```bash
docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/alignment.yaml --step_scheduler.local_batch_size=2
docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/sft.yaml --step_scheduler.local_batch_size=2
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-8gpu.yaml
```

`scripts/run_matrix.sh <container>` runs the whole matrix end to end
(prepare; 1-, 2-, and 8-GPU stages; vLLM probe).

## Artifact chain

```text
base snapshot     artifacts/models/nemotron-nano-4b-bf16/           (pristine; training never writes here)
prepared data     outputs/data/                                     (feature-cache/ + manifest.jsonl + stage slices)
alignment writes  outputs/alignment/projector/                      (mm-projector.safetensors + manifest)
SFT reads it,     outputs/sft/checkpoints/epoch_<K>_step_<N>/model/ (PEFT adapter)
                  outputs/sft/projector/                            (projector, updated — NOT in the PEFT dir)
GRPO reads SFT's projector + adapter,
                  outputs/grpo/checkpoints/step_<N>/policy/weights/ (updated LoRA only)
```

Every artifact is plain Hugging Face format, so nothing needs an export or
conversion step. The SFT step directory also carries optimizer/RNG/dataloader
state beside `model/` for native resume; `model/` itself is a standard PEFT
directory whose `adapter_config.json` records the resolved target modules.
GRPO's layout is NeMo RL's checkpointer (`step_<N>`; `save_period` and
`keep_top_k` in `configs/grpo.yaml`). The GRPO policy loads SFT's projector
export into a sidecar-held projector; the package suite verifies that handoff.
AutoModel's `checkpoints/LATEST` link selects the SFT adapter for GRPO.

## Inference

The trained model is three artifacts — the pristine base snapshot, the SFT
projector export (warm-started from alignment, then jointly trained with the
adapter), and the LoRA adapter — and the base is the one training never
touches. `scripts/infer.py` serves all three together on any image you hand
it: the pinned frozen CLIP encodes the image online (resolved from the HF
cache by repo id and revision, exactly as `llava_example.py` encoded the
training data), the trained projector turns the features into soft tokens, the
adapter is merged into the base weights (the same merge GRPO's refit
performs in memory at every update), and the example's vLLM plugin
generates, with the prompt rendered byte-identically to the GRPO rollout:

```bash
docker exec llava-example \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
  scripts/infer.py --adapter outputs/grpo/checkpoints \
    --input /opt/llava-example/outputs/my-image.jpg \
    --question "How many objects are in the image?"
```

`--adapter` accepts the checkpoints root (the latest step is selected and
printed) or one specific PEFT directory — point it at
`outputs/sft/checkpoints` to compare the SFT adapter against the GRPO one.
The merge is pure safetensors on CPU (~8 GiB for the 4B), no peft or model
instantiation. The `outputs/` mount is host-visible, so dropping an image
there is the easy path into the container. The SFT/GRPO slices were
CLEVR-Math, so counting-style questions (answered "The answer is N") show
the trained behavior best.
minimal engine gate (random embeddings, no trained weights).

## Tests

```bash
pytest examples/llava/tests            # CPU: application contracts
LLAVA_EXAMPLE_HUB_TEST=1 pytest examples/llava/tests -k hub   # opt-in pinned-Hub check
pyrefly check                          # types: package + this example (from the repo root)
```
