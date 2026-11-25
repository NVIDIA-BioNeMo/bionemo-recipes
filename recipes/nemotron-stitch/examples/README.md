# Runnable modality examples

| Example                          | Raw inputs                          | Encoder            | Stages                                       |
| -------------------------------- | ----------------------------------- | ------------------ | -------------------------------------------- |
| [Genetics](genetics/README.md)   | Eight short synthetic DNA sequences | Evo2-20B, layer 12 | Alignment, full decoder SFT, GRPO, inference |
| [Chemistry](chemistry/README.md) | Eight simple SMILES strings         | NV-KERMT-70M-v2    | Alignment, full decoder SFT, GRPO, inference |
| [LLaVA](llava/README.md)         | Images                              | CLIP               | Alignment, LoRA SFT, GRPO, inference         |

Genetics and chemistry share a chat-style JSONL input: `sample_id`, `messages`,
and `answer`. Messages can interleave text blocks with encoder inputs:
`{"type": "dna", "sequence": "ACGT"}` or
`{"type": "molecule", "smiles": "CC"}`. Each block places its soft tokens among
the text; the raw sequence or SMILES is consumed by the encoder. The example
READMEs show complete two-input records. Like HF multimodal chat, the list
order is the prompt order. These custom modalities are encoded by the demo
preparation scripts, which retain typed messages alongside cached features.
Chemistry blocks also carry canonical graph geometry; DNA span lengths come
from the sequence and the configured Evo2 pooling window. The examples define explicit `KERMTProcessor` and `Evo2Processor` classes
that wrap the base checkpoint processor loaded by `AutoProcessor`. They format
only their `molecule` or `dna` blocks in `apply_chat_template` and delegate
native processing to that base instance. Images and other native blocks pass through unchanged. Histories preserve system/user/assistant roles, plain
text content, and text-only turns or records. `answer` is the ground-truth value used for scoring; the default training
response is `\boxed{...}`. Optional `target` supplies a different
completion, `reasoning_content` supplies the native assistant reasoning field,
and `chat_template_kwargs` selects per-record rendering. The bundled first and
last records demonstrate reasoning; the remaining records use direct answers. No custom chat-template engine is needed.

Genetics and chemistry include their raw JSONL inputs, endpoint clients, model
wiring, and production-based training configs. Their count questions exercise the
training path; the same eight records are used in every stage. Training defaults
now follow the production recipes; use bounded step/data overrides for sanity
checks. The bundled records are not evaluation datasets.

Keep genetics and chemistry application Python within a combined budget of
600 nonblank lines, excluding tests. Use package and upstream APIs for training,
configuration, artifacts, and transport.

## Build the training environment

The single [Dockerfile](Dockerfile) builds one image for LLaVA, genetics, and
chemistry. It installs Stitch, all three example plugins, pytest, RDKit, and
the typed BioNeMo perception client into the pinned NeMo framework interpreters.
It does not reinstall training frameworks. The default base uses BioNeMo's public mirror.
From the BioNeMo `recipes/nemotron-stitch` directory:

```bash
docker build -f examples/Dockerfile \
  -t stitch-examples .
```

Select an example with `-w /opt/stitch/examples/llava`,
`-w /opt/stitch/examples/genetics`, or `-w /opt/stitch/examples/chemistry` when
running the image. LLaVA uses local CLIP; genetics and chemistry use encoder
services. The optional `super` build target selects Transformers 5.18 in every
worker and requires a compatible AutoModel base supplied through `BASE_IMAGE`.
The image applies no framework source patches.

The [Python container client](../clients/python/README.md) is bundled in this
recipe and installed from the local build context. Genetics and chemistry
connect to separately deployed encoder services.

For an existing compatible NeMo environment, install this package and the
selected example with `--no-deps`, install `./clients/python[numpy]` from the
recipe directory, and put the selected example directory on `PYTHONPATH`.

## Connect to the encoders

Set the **base URL** or full `/v1/forward` URL for the selected service:

```bash
export EVO2_URL=http://127.0.0.1:8000
export KERMT_URL=http://127.0.0.1:8003
# Optional for authenticated endpoints; omit for local open services:
# export EVO2_TOKEN=...
# export KERMT_TOKEN=...
```

The client sends tokens as bearer authorization headers. Tokens and endpoint
URLs are not written into feature manifests or caches. Preparation fails on
HTTP errors, invalid checksums, wrong layers/pooling, nonfinite features, or
incompatible geometry. It never substitutes random features.

Use the public [Evo 2 perception container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/clara/containers/evo2-embed/-)
for genetics. The 20B profile matches the shipped configs. The first launch
downloads the selected checkpoint into the mounted cache. Run this on a
supported Hopper GPU with enough memory for Evo 2-20B:

```bash
export EVO2_IMAGE=nvcr.io/nvidia/clara/evo2-embed:0.1.0
export EVO2_CACHE_DIR="${EVO2_CACHE_DIR:-$HOME/.cache/bionemo/evo2-embed}"
mkdir -p "$EVO2_CACHE_DIR"
# The container runs as UID 1000. Give it write access to the cache before starting.
docker pull "$EVO2_IMAGE"
docker run --rm --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:8000:8000 \
  -v "$EVO2_CACHE_DIR:/opt/nim/.cache" \
  -e EVO2_MODEL_ID=evo2-20b-1m \
  -e EVO2_EMBEDDINGS_LAYER=decoder.layers.12 "$EVO2_IMAGE"
```

To use the 1B profile, set `EVO2_MODEL_ID=evo2-1b-8k` in the launch command
and follow the [genetics configuration notes](genetics/README.md#setup) to
use 1,920-wide features. Both profiles use layer 12 and window-64 mean pooling
for this recipe.

For chemistry, obtain the URL and any credentials for a compatible KERMT
deployment from your service operator. The service must expose unpooled
molecule, atom, and both bond branches through `/v1/forward`.

The public ESM-3 perception image is
`nvcr.io/nvidia/clara/esm3-embed:0.1.0`. It serves protein embeddings and is
listed here for discovery alongside the Evo 2 image. See the
[ESM-3 container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/clara/containers/esm3-embed/-).

Stop encoder services after cache preparation to free their GPUs for training.

## Prepare and train

Select one example, then open its training shell. Linux host networking lets
this container reach services listening on the host's loopback interface:

```bash
example=genetics  # or chemistry
mkdir -p "examples/$example/artifacts" "examples/$example/outputs"
docker run --rm -it --gpus all --ipc=host --network host --cap-add SYS_PTRACE \
  -e EVO2_URL -e EVO2_TOKEN -e KERMT_URL -e KERMT_TOKEN \
  -v "$PWD/examples/$example/artifacts:/opt/stitch/examples/$example/artifacts" \
  -v "$PWD/examples/$example/outputs:/opt/stitch/examples/$example/outputs" \
  -w "/opt/stitch/examples/$example" stitch-examples
```

Inside the selected example directory, download the base model:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 \
  --local-dir artifacts/models/lightning-30b-bf16
```

Run the preparation command in the [genetics](genetics/README.md#setup) or
[chemistry](chemistry/README.md#setup) guide. Both scripts accept optional
`--encoder-model MODEL_ID` and
`--encoder-revision REVISION` flags. These record your declared encoder identity
in the manifest header and cache contract, so different declarations produce
different cache identities. Omitted values stay unknown (empty strings) and
preparation continues. These settings record metadata; they do not select or verify
the model running at the endpoint.

Preparation makes **eight requests**, one per raw input, then publishes one
manifest with 24 stage-tagged records and separate `manifest-{stage}-train.jsonl`
files. Every cached tensor is checksummed. Generated data lives in
`artifacts/data/`; model weights, caches, and `outputs/` are ignored by Git.
For another preparation run, use `python prepare.py --output artifacts/data-new`
and update config paths, or remove the generated `artifacts/data/` first.
Published caches are immutable. Regenerate older prepared data for either demo: manifests now retain typed
content blocks with cached geometry; chemistry also groups cached features by
branch across all inputs and records `molecule_count`. `--input` accepts another JSONL in the same format.

Training settings follow the selected production recipes in
`nemotron-stitch-demos`: indexed full-parameter chemistry, flanked-v3 shuffled
full-parameter Lightning SFT, and paired public Super. Alignment trains
projectors; SFT trains the decoder and projectors, including native boundary
embeddings. All alignment/SFT configs use THD packing, one pack per microbatch.
Both examples instantiate `PackedManifestFeatureDataset` directly, with the
native packed collator and processor configuration. Chemistry supplies its
three-projector `ChemistryCollator` through the upstream collator loader.

| Recipe                         | Global batch | Updates | Warmup | Peak LR | Pack tokens |
| ------------------------------ | -----------: | ------: | -----: | ------: | ----------: |
| Chemistry alignment            |      8 packs |     705 |     21 |    3e-3 |      16,384 |
| Chemistry full SFT             |      8 packs |   1,410 |     42 |    1e-5 |      16,384 |
| Gene Lightning/Super alignment |     16 packs |   2,000 |     60 |    3e-3 |       4,096 |
| Gene Lightning full SFT        |     16 packs |   5,000 |    250 |    1e-5 |       4,096 |
| Gene paired Super full SFT     |     16 packs |   4,945 |    248 |    1e-5 |       4,096 |

The gene SFT source runs used padded samples; these examples deliberately use
THD packs in both supervised stages. The per-update token exposure changes;
this is a recipe with production optimization settings, not bit-exact replay.
Gene YAMLs directly configure the production Evo2-20B feature width (8192)
and pooling window (64). Preparation reads those same values with
`python prepare.py --config configs/alignment.yaml`. For a local Evo2-1B
endpoint, set the stage YAML projector input widths to 1920. Use a new cache
with matching geometry when changing encoder settings;
span lengths are derived by the processor and are not dataset fields. The Dockerfile
pins the production fused cross-entropy implementation. Each config cites its
source recipe; loss reduction, precision, scheduler, and distributed settings
come from that source. Launch topology lives in the application, including the
paired Super multinode launch. There is no single-node memory-tuning overlay.

### Bundled-data sanity run

The eight short rows pack into fewer than eight training samples. The pinned
AutoModel distributed sampler drops incomplete rank groups, so the original
manifest gives empty loaders on eight ranks; more epochs cannot repair this.

For a bounded sanity run, write a separate repeated view of the prepared
manifest. Unique sample IDs reuse the same checksummed feature keys, so this
makes no additional encoder requests and leaves the published cache intact.
These repeated exposures have no interpretation as additional training data.

```python
import json
from pathlib import Path

source = Path("artifacts/data/manifest.jsonl")
rows = [json.loads(line) for line in source.read_text().splitlines()]
repeated = [
    {**row, "sample_id": f"{row['sample_id']}-sanity-{copy}"}
    for copy in range(256)
    for row in rows
]
source.with_name("manifest-sanity.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in repeated)
)
```

Use this repeated view with the alignment and SFT commands in the
[genetics](genetics/README.md#1-alignment) and
[chemistry](chemistry/README.md#1-alignment) guides.

Keep the `--` prefix on every AutoModel override: the pinned parser silently
ignores bare `key=value` arguments. After changing raw data, tokenizer, pack
budget, or rank count, verify that each distributed loader has at least one
optimizer update's worth of microbatches; the repeated view is tested at the loader boundary
only for the bundled records and documented eight-rank settings. This does
not qualify end-to-end training or model quality.

Production dataset conversion, stage/split selection, cache publication,
qualification launchers, and run receipts belong in the demos repository. A
bounded qualification keeps these example configs and model code, changes the
data/model/output paths, and stops early; it preserves the full LR schedule,
batch sizes, packing, and production node topology. Do not use the tiny bundled
preparation script as an in-memory encoder pipeline for a million-row corpus.

Alignment exports `outputs/alignment/projector/`. SFT loads that artifact and
writes `outputs/sft/projector/`, `outputs/sft/trainability.json`, and decoder
checkpoints under `outputs/sft/checkpoints/`.

## GRPO

GRPO freezes the SFT projectors. Chemistry defaults to the indexed full-decoder
recipe on two eight-H100 nodes: 24 prompts × eight responses, LR 1e-6, KL 0.01,
1,024 new tokens, and vLLM TP2/EP2. Gene defaults to a fresh rank-128 LoRA on the
consolidated full-SFT decoder, using the reference four-node settings:
256 prompts × eight responses, LR 1e-6 with three-update warmup, KL 1e-4,
512 new tokens, policy EP8/DP4, and generation EP8. `grpo-full.yaml` disables
LoRA for gene; for chemistry it is an alias of the default full-decoder recipe.
Super uses the same gene GRPO settings with native Super model wiring; its
end-to-end GRPO qualification remains pending.

```bash
python -m nemotron_stitch.nemo_rl.runner --config configs/grpo.yaml
```

Set `STITCH_SFT_MODEL` to select another consolidated SFT decoder. Both examples
use NeMo RL's built-in `geo3k` boxed-answer verifier with `format_score: 0.0`:
only final-answer correctness contributes reward. An MC record supplies its
option letter as `answer`, and the model emits `\boxed{A}`. No reasoning,
formatting, or semantic reward bonus is added. This also scores the bundled
integer-count questions. Explicit production `target`/reasoning fields remain
unchanged. Policy sequence packing stays disabled, matching the selected GRPO
recipes; THD packing applies to alignment and SFT.

The eight-row data cannot supply the production rollout batch. For a bounded
sanity GRPO run, override the prompt count, global batch, duration, and epoch
limit explicitly. Qualification on production data keeps the production batch
and node topology. For the shipped topology, join the nodes to the same Ray
cluster before launching the runner once on the head node. Use the same image and make the model,
prepared data, and SFT artifacts available at identical paths on every node.
See [NeMo RL's cluster setup](https://docs.nvidia.com/nemo/rl/latest/cluster.html).

## Inference with prepared DNA or SMILES features

The genetics and chemistry inference commands use the same dataset processor,
frozen SFT projectors, and vLLM plugin as GRPO rollouts. The tokenizer comes
from the original model snapshot. The helper generates from the first
prepared GRPO record by default. Use `--record-index` to select another row or
`--tensor-parallel-size` to match your GPU allocation and model memory requirements.

The helper uses vLLM's [Python generation API](https://docs.vllm.ai/en/stable/features/multimodal_inputs/#embedding-inputs)
to send the custom modality's projected tensors with the rendered prompt. To
use a new input, prepare it with the matching perception endpoint and point the
config's manifest and cache paths at that prepared data. For an HTTP service,
the application needs to run the same input preparation and projection before
calling vLLM.
