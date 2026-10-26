# LangChain docs snapshot

A frozen copy of 41 pages of the LangChain / LangGraph Python documentation, used as the
retrieval corpus for the benchmark agent. Benchmarks read this snapshot, never the live site,
so results are reproducible.

- Source: https://docs.langchain.com/oss/python/ (pages listed in `pages.txt`)
- Upstream repository: https://github.com/langchain-ai/docs
- License: MIT, Copyright (c) 2025 LangChain. See `LICENSE` in this folder.
- Fetched with LangChain's `WebBaseLoader`, keeping only the page's main content area, then
  stripped of page chrome (`clean()` in `bench/corpus/fetch_langchain_docs.py`).
- `snapshot.jsonl`: one page per line with `path`, `url`, `fetched_at`, `sha256`, `chars`, `text`.

Refresh or compare with the live site:

```bash
python -m bench.corpus.fetch_langchain_docs           # rewrite snapshot.jsonl
python -m bench.corpus.fetch_langchain_docs --check   # list pages that changed upstream
```

Text is extracted from HTML, so some headings and table cells run together without spaces.
Code blocks are preserved.
