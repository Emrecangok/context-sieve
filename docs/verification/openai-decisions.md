# OpenAI Decisions API verification notes (2026-10-26)

Primary decider for v0.1 (no Jev key; no local GPU toolchain).
Sources: the official guide (developers.openai.com/api/docs/guides/decisions) and
`openai-python` source. The SDK signature was also checked locally (openai 3.37.0).
**No live call has been made yet** (no `OPENAI_API_KEY` on this machine).

## Verified

- SDK: `client.decisions.create(*, input, model, questions, prompt_cache_key=, prompt_cache_options=, safety_identifier=)`
  → `Decision`, first shipped in `openai` 3.26.0 (2026-10-06). Endpoint `POST /v1/decisions`.
- Model: `gpt-6-luna` is the only supported model.
- `input`: a string, or an array of items: `message` (roles `user` / `developer` / `system` / `assistant`),
  `function_call` (`call_id`, `name`, `arguments`), `function_call_output` (`call_id`, `output`),
  `custom_tool_call`, `custom_tool_call_output`. This maps one-to-one onto LangChain messages.
- `questions`: 1–400 per request. Types `predicate` (`instructions`), `choice` (`choices`: 2–255,
  value str|bool), `score` (`levels`: 2–10). `name` is optional; must be unique.
- Response: `answers[]` in question order. predicate → `probability`; choice → `choice`, `probabilities[]`,
  `confidence`; score → `score` (expected 0-based level index), `probabilities[]`, `confidence`;
  `refusal` → no value (we map it to `p_yes=None` → keep).
- `usage`: `input_tokens`, `input_tokens_details.cached_tokens`, `cache_write_tokens`.
- Pricing: $0.10 / 1M uncached input tokens. Output tokens are not charged. Cache hits are discounted;
  cache writes are charged. Prompt caching needs a shared prefix of ≥ 1,024 tokens.
- The guide calls the probabilities "estimates" and recommends setting thresholds from labelled
  examples. Calibration and determinism are not documented.

## Not verified

- Context window / max input tokens for `/v1/decisions` (aggregators claim ~1M for gpt-6-luna; not
  stated in the guide).
- Rate limits.
- Allowed characters in question `name`. We send positional names `q0..qN` to avoid this.
- Confidence formula for choice/score.
- Run-to-run stability of `probability` (measure: repeat requests and report the spread).

## Design consequences

- One request carries the full history once, plus every candidate's predicates (≤ 400). Messages are
  tagged `[m{i}]` so a question can point at one message. Tool outputs are also addressable by `call_id`.
- Repeated pruning passes over a growing history share a prefix → set `prompt_cache_key`.

## Laya

Considered and dropped by the user (2026-10-26). For the record: its base checkpoints are near chance
zero-shot (0.36 on its own typed-decisions suite) and have a 512–1024 token context.
