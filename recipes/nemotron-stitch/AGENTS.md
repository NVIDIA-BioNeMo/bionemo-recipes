# AGENTS.md

Instructions for agents and humans working in this recipe. The BioNeMo
repository's root `AGENTS.md` also applies.

## The one rule

**Write as little code as possible.**

This package exists to remove the friction specific to adding a modality to
NeMo AutoModel, NeMo RL, and vLLM. Using it should still feel like using those
frameworks directly. It fails at its purpose if it becomes another framework
layer over them. Every line here is a liability we have chosen to accept, and
the measure of a good change is usually how much code it deletes.

Concretely, before writing anything, in this order:

1. **Does an upstream framework already do it?** NeMo AutoModel, NeMo RL, vLLM,
   PEFT, Transformers, and PyTorch collectively do most of this. Use the public
   API even when it is slightly awkward. An awkward call site is cheaper than an
   owned implementation.
2. **Is the pain specific to adding a modality?** The number or identity of
   current users is not the bar. The change belongs here when it removes
   modality-agnostic friction in connecting an encoder, projector, or soft
   tokens to the upstream frameworks. General training, inference,
   configuration, and orchestration features belong upstream or in the
   application.
3. **Could the framework own it eventually?** Framework integration code should
   have a plausible destination in AutoModel, NeMo RL, or vLLM. Keep the seam
   thin enough to delete when that upstream capability lands. Package-owned
   contracts such as artifacts and index geometry may remain here.
4. **Does an existing user already do it?** ct-nemotron and genome-research are
   the original source material, not an exhaustive list of consumers. Search
   current users and move a proven, modality-agnostic implementation when one
   exists; do not write another version.
5. **Can configuration do it?** An upstream config key beats a package parameter,
   which beats a hook, which beats a subclass. Do not introduce a parallel
   lifecycle, configuration system, or abstraction for a framework concept.
6. **Only then**, write it — and if it is a framework workaround, record it per
   the next section.

Corollaries:

- No implementation under `src/nemotron_stitch/` may encode behavior,
  assumptions, or APIs specific to one modality. Neutral naming does not make
  modality-specific code generic. If swapping the encoder for a different one
  *in the same modality* would change a file, it is domain code. Modality names
  may appear only in docstrings and tests.
- `examples/` is the deliberate exception to that corollary: an example is
  domain code by construction, names its modality, and is held to the
  code-deletion gate in its plan instead. The `src/` rule itself is
  unchanged.
- No framework import at package import time. The base wheel must import cleanly
  in an image with no vLLM and Transformers pinned at 4.48.1. Use a lazy import
  with a sentinel fallback.
- Do not add a dependency to satisfy one function. Copy the twelve lines, or do
  without.
- Do not add a compatibility shim for a version we do not pin.

## Track the upstream fixes that would simplify us

Most of the code here that is ugly is ugly because a framework has no hook for
what we need. That is worth tracking, because each one is a small upstream PR
away from being deletable — and because in six months nobody will remember which
awkward-looking function is load-bearing and which is routing around a bug that
has since been fixed.

The tracker is [`docs/upstream-gaps.md`](docs/upstream-gaps.md).
Its first table is the modality-integration upstream roadmap; a separate table
keeps adjacent framework findings from being mistaken for package scope. Its
`U-` identifiers are stable and are cited by workaround comments.

So:

- **When you work around a framework, add a row to `upstream-gaps.md`, then
  cite its `U-` identifier in the code comment.** The comment says why, names
  the pinned revision, and says what would let us delete it.
- **Ordinary code does not need a comment like this.** The projector zoo, the
  artifact schema, the feature cache — those are our contracts, not workarounds.
- **When an upstream fix lands, delete our version.** Do not add a branch for
  both. That is the whole point of keeping the list.

The shape, from this repository:

```python
def register_models(architecture, model_cls, registry=None):
    """Register ``model_cls`` under ``architecture`` in AutoModel's model registry, once."""
    # NeMo AutoModel 24b47e856263d313b942f0ed666c63fff83306b4 resolves
    # architectures through a private _transformers.registry and has no public
    # out-of-tree architecture registration hook. This makes the private import
    # idempotent and fails closed on conflicting registrations. Delete after
    # adopting U-2.
    from nemo_automodel._transformers.registry import ModelRegistry

    # ... register, with duplicate-identical tolerated and conflict raising
```

Two things make that comment useful rather than decorative: it names the pinned
revision, so a pin bump is a prompt to recheck it; and it says what upstream
would have to expose, so the person deleting it knows what they are waiting for.
If you cannot say what would let us delete it, it probably is not a workaround —
it is just our code, and it does not need the comment.

## Changes

- **One concern per PR.** Do not batch unrelated changes. The phased port plans
  describe the original extraction; they do not define the current user base or
  gate new work.
- **Refactors are numerically neutral and gated on it.** Keep the relevant
  bit-exact forward-parity fixtures green. A change that cannot is not a
  refactor and needs its own discussion.
- **A contract change is a schema change.** Touching the forward kwargs, the
  index geometry, or the manifest means bumping `SCHEMA_VERSION` in
  `contracts.py` and shipping the converter in the same change.
- **Say what you did not do.** If a phase is partly blocked, finish the rest and
  state plainly what was left and why.

## Testing

Run the suite inside the fully provisioned example image, currently
`svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64`. The seam
guards and framework mixins are the point of the package, and they only execute
against the pinned frameworks the image carries; a bare-metal `pytest`
import-skips them. When the image tag advances, this section follows it.

```bash
docker run --rm \
  -v "$PWD:/opt/nemotron-stitch" \
  -w /opt/nemotron-stitch \
  -e PYTHONPATH=/opt/nemotron-stitch/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64 \
  /opt/nemo_rl_venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

The driver venv covers NeMo RL. Full coverage needs the two worker venvs,
which carry AutoModel and vLLM respectively but not pytest — install it into
the throwaway container first:

```bash
uv pip install -q pytest \
  --python /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
uv pip install -q pytest \
  --python /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
# then run the same pytest command with each of those interpreters
```

On a GPU-less host the CUDA-only kernel tests skip; that is expected. The
GitHub `ci.yaml` job remains the CPU-only PR gate — it proves the base wheel
imports without frameworks, not that the seams hold.

## Vocabulary

This is the LLaVA recipe generalized past vision, and it uses LLaVA's words. See
design doc §1.5 for the historical mapping from the original consumers' names.

- **projector** — the trainable encoder→LM module. Not "adapter".
- **adapter** — LoRA, and only LoRA.
- **soft tokens** — the projector's output, `[N, H_lm]`.
- **features** — the encoder's output, `[N, mm_hidden_size]`.
- **encoder** — the frozen upstream model (LLaVA's "vision tower").

Naming that leaks a modality (`dna_`, `ct_`, `image_`) does not belong here.
Naming that invents a term where LLaVA has one is a review comment.

## Style

Match the surrounding code. Beyond that:

- Type annotations on public signatures. `from __future__ import annotations`.
- Fail closed. An unsupported topology, an unverified provenance value, or an
  ambiguous payload raises; it does not warn and continue. A silently wrong
  scatter produces a plausible loss curve.
- Validate at construction, not per step. Checks belong in the collator and the
  dataset, not in the forward pass or the generation call.
- Comments explain why, not what. A comment restating the code is noise; a
  comment naming the upstream revision that made the code necessary is worth its
  space.
