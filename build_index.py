#!/usr/bin/env python3
"""Build a compact SQLite/FTS5 index from a jawiki XML dump."""

from __future__ import annotations

import argparse
import bz2
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from lxml import etree

try:
    import mwparserfromhell
except ImportError:  # pragma: no cover - dependency is listed in requirements
    mwparserfromhell = None


WIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
DEFAULT_MAX_CHARS = 12000


def child_text(element: etree._Element, name: str) -> str:
    child = element.find(f"{WIKI_NS}{name}")
    return (child.text or "") if child is not None else ""


def clean_wikitext(source: str, max_chars: int) -> str:
    """Turn wikitext into a readable lead suitable for search and display."""
    # The full parser is more accurate but too slow for millions of pages.
    # Opt into it for a smaller/custom dump with WIKI_USE_MWPARSER=1.
    if mwparserfromhell is not None and os.getenv("WIKI_USE_MWPARSER") == "1":
        code = mwparserfromhell.parse(source)
        text = code.strip_code(normalize=True, collapse=True)
    else:
        text = source
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
        text = re.sub(r"<nowiki.*?</nowiki>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        for _ in range(3):
            text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
        text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"^\s*=+.*?=+\s*$", " ", text, flags=re.M)
        text = re.sub(r"'{2,5}", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def iter_pages(dump_path: Path, max_chars: int) -> Iterator[tuple[str, str]]:
    with bz2.open(dump_path, "rb") as stream:
        context = etree.iterparse(stream, events=("end",), tag=f"{WIKI_NS}page", huge_tree=True)
        for _, page in context:
            namespace = child_text(page, "ns")
            redirect = page.find(f"{WIKI_NS}redirect")
            if namespace == "0" and redirect is None:
                title = child_text(page, "title").strip()
                revisions = page.findall(f"{WIKI_NS}revision")
                if title and revisions:
                    source = child_text(revisions[-1], "text")
                    # Search uses the lead, so avoid repeatedly cleaning very
                    # large tails of articles that will be discarded anyway.
                    source = source[: max_chars * 5]
                    text = clean_wikitext(source, max_chars)
                    if text:
                        yield title, text
            page.clear()
            while page.getprevious() is not None:
                del page.getparent()[0]


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript(
        """
        DROP TABLE IF EXISTS pages_fts;
        DROP TABLE IF EXISTS pages;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE pages (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            url TEXT NOT NULL
        );
        -- Trigram search works well for Japanese text without requiring MeCab.
        -- The FTS index is contentless to avoid storing a second copy of text.
        CREATE VIRTUAL TABLE pages_fts USING fts5(
            title, text, content='', tokenize='trigram'
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX pages_title_idx ON pages(title);
        """
    )
    return connection


def build(dump_path: Path, database_path: Path, max_chars: int) -> None:
    if not dump_path.exists():
        raise FileNotFoundError(f"ダンプが見つかりません: {dump_path}")
    connection = create_database(database_path)
    count = 0
    started = time.monotonic()
    page_batch: list[tuple[int, str, str, str]] = []
    try:
        for title, text in iter_pages(dump_path, max_chars):
            count += 1
            url = "https://ja.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()")
            page_batch.append((count, title, text, url))
            if count % 5000 == 0:
                connection.executemany(
                    "INSERT INTO pages(id, title, text, url) VALUES (?, ?, ?, ?)",
                    page_batch,
                )
                page_batch.clear()
                connection.commit()
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"\r{count:,} pages ({count / elapsed:.0f}/s)", end="", flush=True)
        if page_batch:
            connection.executemany(
                "INSERT INTO pages(id, title, text, url) VALUES (?, ?, ?, ?)",
                page_batch,
            )
            page_batch.clear()
        connection.commit()
        print(f"\nFTS5索引を構築中: {count:,} pages", flush=True)
        connection.execute("INSERT INTO pages_fts(rowid, title, text) SELECT id, title, text FROM pages")
        connection.execute("INSERT INTO meta VALUES ('page_count', ?)", (str(count),))
        connection.execute("INSERT INTO meta VALUES ('max_chars', ?)", (str(max_chars),))
        connection.execute("INSERT INTO meta VALUES ('created_at', ?)", (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
        connection.commit()
    finally:
        connection.close()
    print(f"\n完了: {count:,} pages -> {database_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("data/jawiki-latest-pages-articles-multistream.xml.bz2"))
    parser.add_argument("--db", type=Path, default=Path("data/jawiki.sqlite3"))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()
    build(args.dump, args.db, args.max_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
