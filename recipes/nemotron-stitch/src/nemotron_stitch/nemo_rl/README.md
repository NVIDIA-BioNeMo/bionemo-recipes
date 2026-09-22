# `nemo_rl/`

NeMo RL integration: the seams that carry encoder features and soft tokens
through GRPO — policy training, rollout transport, and vLLM generation.

- `data.py` — dataset over modality-neutral packed encoder tensors, with
  collator-side validation (sentinel counts, projected shapes) and the
  chat-template-kwargs tokenizer proxy.
- `policy.py` — DTensor policy worker bridge for arbitrary encoder payloads,
  with the donor-adapter provenance gate over NeMo RL's own compact-PEFT warm
  start.
- `runner.py` — owned GRPO bootstrap: registers the worker runtimes behind
  NeMo RL's config-driven worker-extension FQNs (U-5).
- `transport.py` — modality-neutral callback helpers moving encoder payloads
  between data and policy planes.
- `vllm_worker.py` — vLLM generation worker: plugin registration,
  generation-only `hf_overrides` composition, and the resident/offload
  lifecycle policy.

Framework imports are lazy; the base wheel never imports NeMo RL.

## Upstream gaps

Subset of `docs/upstream-gaps.md`
(the single source of truth). Of this subpackage's modality-roadmap items,
U-4, U-5, and U-8 are adopted upstream; U-18 remains open, and U-9, U-20, and
U-21 are recorded there as adjacent work. U-19 disappeared with the U-4 shim.
