"""Embed the docs snapshot with OpenAI and save the index to data/ (needs OPENAI_API_KEY).

    python -m bench.build_index
"""

from __future__ import annotations

from bench.docs_agent import INDEX, build_index, embed_model_name, load_env, openai_embeddings


def main() -> int:
    load_env()
    store = build_index(openai_embeddings())
    print(f"indexed {len(store.store)} chunks with {embed_model_name()} -> {INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
