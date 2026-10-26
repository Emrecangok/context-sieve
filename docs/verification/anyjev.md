# AnyJev verification notes (2026-10-26)

Read from the source (`nokia-applied-research/AnyJev@main`) and PyPI metadata.
Nothing was run locally yet.

## Facts that correct the project brief

| Brief said | Source says |
|---|---|
| v0.2.0 | PyPI latest is **0.3.0** (2026-10-06) |
| Levels L0 / L1 / L2, `save_artifacts` | 0.3.0 has only `raw` and `L0`. L1/L2 (label-trained) were removed; they remain in `anyjev==0.2.0`. No artifact save/load. |
| Answer-mass gate `m >= 0.99` | `diagnostics["answer_mass"]` is computed but never gated, warned or raised on. **We must gate it ourselves.** |
| Rotation gives no accuracy gain for 2-option questions | Model-dependent. Qwen3-4B on `injection` (K=2): raw 0.683 → L0 0.797, flip 0.203 → 0.000. Measure raw vs L0 per model. |

## API surface we depend on

- `Question.noul(text, name=None)` → labels `("Yes", "No")`; `Question.choice`, `Question.score`.
- `Decider(backend, level="L0", prior="batch", ...)`
  - `decide(state, questions) -> DecisionSet`
  - `decide_batch(states, question) -> list[Decision]` (one question, many states; the batch prior is estimated here)
- `Decision`: `probs`, `level`, `diagnostics`, `answer`, `confidence` (= max prob), `p_true` (noul only).
- Backends: `HFBackend(model_name, device="cuda", dtype="bfloat16", batch_size=16, **model_kwargs)`
  (no truncation — prompt length is our job), `LlamaCppBackend(model_path, n_ctx=4096, n_gpu_layers=0, ...)`
  (in-process llama-cpp-python, default Windows wheel is CPU-only), vLLM, SGLang, and `backends/fake.py` for tests.
- `Tacit.from_pretrained(repo, engine="transformers", dtype="bfloat16", device_map="auto")`,
  `decide(state, question, options=None, kind="choice"|"yes_no"|"score")` → dict with `answer`, `probs`, `margin`, `route`.
  Different shape from `Decider`; needs its own adapter.

## Pitfall for pruning

`docs/limitations.md`: the L0 batch prior "costs accuracy when one label dominates a batch;
use `prior="content_free"` or `prior="none"`". In pruning most turns are irrelevant, so "No"
dominates. Default to `prior="content_free"` and include `prior` in the ablation.

## Model choice for a 6 GB GPU (RTX 3050 Laptop)

Tacit checkpoints (HF `morriszjm/*`), JevBench public, single forward pass:

| Checkpoint | Base | Acc | bf16 weights | Fits 6 GB? |
|---|---|---|---|---|
| Tacit-1.7B | Qwen3-1.7B | 143/231 = 0.619 | ~3.4 GB | yes |
| Tacit-4B | Qwen3-4B | 167/231 = 0.723 | ~8 GB | only quantized (unverified) |
| Tacit-8B | Qwen3-8B | 170/231 = 0.736 | ~16 GB | no |

Status: deferred. The user's machine has no CUDA toolchain, so v0.1 uses the OpenAI Decisions API
(see `openai-decisions.md`). When local models are added: start with Qwen3-1.7B, training-free
`Decider` (raw, L0) vs `Tacit-1.7B` (same base, distilled).
