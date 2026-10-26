"""Fetch the LangChain docs pages listed in pages.txt and write a snapshot.

    python -m bench.corpus.fetch_langchain_docs            # refresh snapshot.jsonl
    python -m bench.corpus.fetch_langchain_docs --check    # compare live pages with the snapshot

The live site changes daily, so benchmarks read the committed snapshot, not the site.
Pages are loaded with LangChain's WebBaseLoader, keeping only the main content area
(no navigation, sidebar or footer).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

BASE_URL = "https://docs.langchain.com/oss/python/"
HERE = Path(__file__).parent / "langchain_docs"
PAGES = HERE / "pages.txt"
SNAPSHOT = HERE / "snapshot.jsonl"


def page_paths() -> list[str]:
    lines = PAGES.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def load_pages(paths: list[str]):
    os.environ.setdefault("USER_AGENT", "context-sieve-bench (+https://github.com/Emrecangok/context-sieve)")
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(
        web_paths=[BASE_URL + p for p in paths],
        bs_kwargs={"parse_only": bs4.SoupStrainer(id="content-area")},
        raise_for_status=True,
        requests_per_second=2,
        show_progress=False,
    )
    return loader.load()


FOOTER_MARKER = "Connect these docs to your agent of choice via MCP"


def clean(text: str) -> str:
    """Strip page chrome that sits inside the content area."""
    text = text.replace("​", "").replace("Copy page", "")
    cut = text.find(FOOTER_MARKER)
    if cut != -1:
        text = text[:cut]
    return text.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report pages that differ from the snapshot")
    args = parser.parse_args(argv)

    paths = page_paths()
    docs = load_pages(paths)
    if len(docs) != len(paths):
        print(f"expected {len(paths)} pages, got {len(docs)}", file=sys.stderr)
        return 1

    texts = [clean(d.page_content) for d in docs]
    if args.check:
        old = {r["path"]: r["sha256"] for r in map(json.loads, SNAPSHOT.read_text(encoding="utf-8").splitlines())}
        changed = [p for p, t in zip(paths, texts) if old.get(p) != sha256(t)]
        print(f"{len(changed)} of {len(paths)} pages differ from the snapshot", *changed, sep="\n  ")
        return 0

    fetched_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    empty = [p for p, t in zip(paths, texts) if len(t) < 500]
    if empty:
        print("pages with no content (layout changed?):", *empty, sep="\n  ", file=sys.stderr)
        return 1
    with SNAPSHOT.open("w", encoding="utf-8", newline="\n") as f:
        for path, doc, text in zip(paths, docs, texts):
            record = {
                "path": path,
                "url": doc.metadata.get("source", BASE_URL + path),
                "fetched_at": fetched_at,
                "sha256": sha256(text),
                "chars": len(text),
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(texts)} pages ({sum(map(len, texts)):,} chars) to {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
