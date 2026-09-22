# AGENTS.md

Instructions for agents and humans working in this repository.

## Goal

When adding demos, build lightweight, end-to-end Nemotron Stitch demonstrations
such as:

- `nemotron-rna/`
- `nemotron-kermt/`

Use [`llava/`](llava/) as the reference for their scope. Each demo should contain only the domain code
and configuration needed to understand and reproduce a competitive recipe. It
should teach a reader how to build their own model, not grow into a reusable
framework of its own.

## Optimize for a small, instructive recipe

- Write as little code as the end-to-end recipe permits. Prefer direct use of
  public APIs from Nemotron Stitch and its upstream frameworks over wrappers,
  indirection, or locally invented abstractions.
- Include a component only when it is required by the model contract, makes the
  example meaningfully easier to understand, or has a credible effect on final
  quality. Avoid speculative flexibility and research surfaces that are not
  exercised by the shipped recipe.
- Keep the complete learning path visible: data preparation, model and modality
  integration, alignment, SFT, evaluation, and RL or inference where applicable.
  Match the LLaVA example's breadth without copying features the domain does not
  need.
- Prefer one clear qualified path over several partially supported alternatives.
  Pin and record the model, data, framework revisions, and defaults used to
  obtain reported results.
- Treat configuration as part of the explanation. Defaults should be runnable,
  realistic, and backed by either a qualified run or a clearly labeled
  smoke-test purpose.

## Explanatory-code standard

Readers are expected to use this code as a blueprint for their own models.
Document accordingly, including private helpers.

- Give every module, class, and non-trivial function a docstring. For private
  functions, explain why the helper exists, which boundary or invariant it
  owns, and why the logic is not delegated to Stitch or an upstream framework.
- Document shapes, units, token/span geometry, trainability, lifecycle, and
  failure behavior wherever those facts are part of the contract.
- Comments should explain non-obvious design choices and constraints, not
  restate individual lines of code.
- Use descriptive domain vocabulary consistently. Avoid generic helpers whose
  name hides an important RNA- or KERMT-specific assumption.
- Keep READMEs end-to-end: state what the demo owns, what it deliberately omits,
  the qualified configuration and result, exact stage handoffs, and commands a
  reader can run.
- Tests are explanatory artifacts too. Name fixtures and assertions after the
  contract they demonstrate, and cover silent-failure risks such as token
  geometry, feature alignment, masking, trainability, and artifact handoffs.

Obvious one-line accessors do not need essay-length prose, but brevity must not
hide why domain-specific or framework-facing code is necessary.

## Keep the two demos in sync

The demos must not import from one another. Each one should remain readable,
installable, testable, and copyable on its own; intentional duplication is
preferred to a shared demo framework.

Within that boundary, keep analogous choices aligned:

- directory layout, filenames, stage boundaries, CLI style, and output layout;
- configuration structure, key ordering, naming, and model indirection;
- optimizer and scheduler choices, default learning rates, warmup policy,
  training duration, checkpoint policy, logging cadence, and validation shape;
- terminology, docstring style, error handling, and test organization.

When changing one demo, inspect the corresponding code and configs in the
other. Apply an analogous change when it is valid. A domain-specific divergence
is acceptable, but document the reason next to the differing default and in the
relevant README when it affects how a run should be interpreted. Do not force
cosmetic consistency when sequence lengths, data volume, encoder behavior,
memory use, convergence, or evaluation methodology justify a difference.

## Framework boundary: stop rather than patch locally

This repository owns the RNA and KERMT applications. It does not own fixes or
compatibility layers for Nemotron Stitch, NeMo AutoModel, NeMo RL, vLLM,
Transformers, or other shared dependencies.

If the recipe requires a monkey patch, private-framework workaround,
compatibility shim, vendored framework code, or change to a shared integration
seam, **do not implement it in either demo**. Pause that part of the work and
raise the missing capability for implementation in the recipe root (`..`).

The escalation must include:

1. the affected framework and pinned revision;
2. the smallest reproducible failure or missing public hook;
3. the behavior the demo needs and the domain-neutral seam Stitch should expose;
4. what temporary demo code would otherwise have been required; and
5. a corresponding entry in
   [`../docs/upstream-gaps.md`](../docs/upstream-gaps.md),
   following that repository's instructions.

Do not resume the blocked portion by hiding the workaround in a helper. Resume
once the fix is available through a pinned Stitch/upstream revision. Continue
independent, unblocked demo work and report plainly what remains blocked.

Domain behavior belongs here when it is genuinely specific to RNA or KERMT:
data normalization, encoder semantics, token geometry, task losses or rewards,
and domain evaluation. General feature transport, artifact, training,
generation, registration, or distributed-runtime machinery belongs in Stitch
or upstream.

## Change and verification discipline

- Make focused changes and preserve a runnable end-to-end path.
- Fail closed on ambiguous records, unsupported geometry, mismatched artifacts,
  or unverified provenance. A plausible but incorrect training run is worse than
  an early error.
- Validate invariants during data preparation or construction when possible,
  rather than adding repeated checks to the hot path.
- Run **all tests and all training inside the pinned NeMo RL container stack
  declared by the demo's `Dockerfile`**. Normally use the built demo image; the
  exact `BASE_IMAGE` with the demo and Stitch sources mounted is also acceptable
  for unit tests. This includes fast tests: do not use a host Python environment
  as evidence that a change works. Use the driver, AutoModel-policy-worker, or
  vLLM-generation-worker interpreter appropriate to the seam being tested.
  Host-side static checks such as `git diff --check` are fine, but they are not
  substitutes for the container test run.
- Keep the container commands needed to reproduce tests and training in the
  demo README. Do not silently install or patch framework packages in a running
  container; change the pinned image or escalate the missing dependency through
  the framework-boundary process above.
- Smoke tests do not substantiate performance claims.
- Do not claim competitive performance without recording the dataset/split,
  hardware, training budget, checkpoint, metric definition, and observed
  result. Distinguish reproduced results from targets and expectations.
- Before finishing a change, compare the two demos for unintended drift and say
  which verification was run, what was not run, and why.
