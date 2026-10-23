#!/usr/bin/env python3
"""Fast local Japanese Wikipedia search app."""

from __future__ import annotations

import argparse
import html
import os
import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
DB_PATH = Path(os.getenv("WIKI_DB", "data/jawiki.sqlite3"))
MODEL_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
RERANK_TEXT_CHARS = int(os.getenv("RERANK_TEXT_CHARS", "2000"))


class SearchEngine:
    def __init__(self, database: Path, use_model: bool = True) -> None:
        if not database.exists():
            raise FileNotFoundError(f"索引がありません: {database}. build_index.py を実行してください")
        self.database = database
        self.model = None
        self.embedding_cache: OrderedDict[int, object] = OrderedDict()
        if use_model:
            self._load_model()

    def _load_model(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        kwargs = {
            "model_kwargs": {"torch_dtype": dtype},
            "config_kwargs": {"vision_config": None, "audio_config": None},
        }
        try:
            self.model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs)
        except Exception:
            # Older sentence-transformers versions may not understand config_kwargs.
            self.model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", model_kwargs={"torch_dtype": dtype})

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _fts_query(query: str) -> str:
        # SQLite's trigram tokenizer needs explicit 3-character fragments for
        # Japanese text. OR gives the semantic reranker a useful candidate set.
        compact = re.sub(r"\s+", "", query.replace('"', " ").strip())
        fragments = [compact[i : i + 3] for i in range(max(len(compact) - 2, 0))]
        if not fragments:
            return '"' + compact + '"'
        return " OR ".join(f'"{fragment}"' for fragment in fragments)

    def _candidates(
        self,
        query: str,
        limit: int,
        title_contains: str = "",
        refine: str = "",
    ) -> list[sqlite3.Row]:
        with self._connection() as connection:
            title_filter = title_contains.strip()
            refine_filter = refine.strip()
            if len(query) < 3:
                rows = connection.execute(
                    """
                    SELECT id, title, text, url, 0.0 AS lexical_score FROM pages
                    WHERE title LIKE ? AND (? = '' OR title LIKE ?)
                      AND (? = '' OR text LIKE ? OR title LIKE ?)
                    LIMIT ?
                    """,
                    (f"%{query}%", title_filter, f"%{title_filter}%", refine_filter,
                     f"%{refine_filter}%", f"%{refine_filter}%", limit),
                ).fetchall()
            else:
                try:
                    match_sql = "pages_fts MATCH ?"
                    match_params = [self._fts_query(query)]
                    refine_sql = ""
                    refine_params: list[str] = []
                    if refine_filter:
                        if len(refine_filter) < 3:
                            refine_sql = " AND (p.title LIKE ? OR p.text LIKE ?)"
                            refine_params.extend([f"%{refine_filter}%", f"%{refine_filter}%"])
                        else:
                            match_sql += " AND pages_fts MATCH ?"
                            match_params.append(self._fts_query(refine_filter))
                    rows = connection.execute(
                        f"""
                        SELECT p.id, p.title, p.text, p.url, bm25(pages_fts, 10.0, 1.0) AS lexical_score
                        FROM pages_fts JOIN pages AS p ON p.id = pages_fts.rowid
                        WHERE {match_sql}{refine_sql} AND (? = '' OR p.title LIKE ?)
                        ORDER BY lexical_score LIMIT ?
                        """,
                        (*match_params, *refine_params, title_filter, f"%{title_filter}%", limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            return rows

    def _document_embeddings(self, rows: list[sqlite3.Row]):
        import numpy as np

        missing = []
        for row in rows:
            if row["id"] not in self.embedding_cache:
                missing.append(row)
        if missing:
            documents = [
                f"title: {row['title']} | text: {row['text'][:RERANK_TEXT_CHARS]}"
                for row in missing
            ]
            encode_kwargs = {
                "normalize_embeddings": True,
                "truncate_dim": MODEL_DIM,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            try:
                encoded = self.model.encode(documents, batch_size=8, **encode_kwargs)
            except RuntimeError as exc:
                # Other GPU workloads can temporarily reduce available VRAM.
                # Retry with a smaller batch instead of failing the search.
                if "out of memory" not in str(exc).lower():
                    raise
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                encoded = self.model.encode(documents, batch_size=2, **encode_kwargs)
            for row, vector in zip(missing, encoded):
                self.embedding_cache[row["id"]] = vector.astype("float32", copy=False)
                self.embedding_cache.move_to_end(row["id"])
            while len(self.embedding_cache) > 2000:
                self.embedding_cache.popitem(last=False)
        return np.vstack([self.embedding_cache[row["id"]] for row in rows])

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 100,
        title_contains: str = "",
        refine: str = "",
    ) -> list[dict]:
        results, _ = self.search_timed(
            query, limit=limit, candidates=candidates, title_contains=title_contains, refine=refine
        )
        return results

    def search_timed(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 100,
        title_contains: str = "",
        refine: str = "",
    ) -> tuple[list[dict], float]:
        started = time.perf_counter()
        query = query.strip()
        if not query:
            return [], 0.0
        rows = self._candidates(
            query, max(limit, candidates), title_contains=title_contains, refine=refine
        )
        if not rows:
            return [], (time.perf_counter() - started) * 1000
        raw_lexical = [max(-float(row["lexical_score"]), 0.0) for row in rows]
        maximum_lexical = max(raw_lexical, default=0.0)
        lexical = [score / maximum_lexical if maximum_lexical else 0.0 for score in raw_lexical]
        if self.model is not None:
            import numpy as np

            query_vector = self.model.encode(
                f"task: search result | query: {query}",
                normalize_embeddings=True,
                truncate_dim=MODEL_DIM,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            semantic = np.asarray(self._document_embeddings(rows)) @ query_vector
            combined = [0.78 * float(score) + 0.22 * lex for score, lex in zip(semantic, lexical)]
            order = sorted(range(len(rows)), key=lambda i: combined[i], reverse=True)[:limit]
        else:
            order = list(range(min(limit, len(rows))))
            combined = lexical
        results = [
            {
                "title": rows[i]["title"],
                "text": rows[i]["text"],
                "url": rows[i]["url"],
                "score": round(float(combined[i]), 5),
            }
            for i in order
        ]
        return results, (time.perf_counter() - started) * 1000


def make_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(title="日本語Wikipedia高速検索", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": engine.model is not None, "database": str(engine.database)}

    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(10, ge=1, le=50),
        title: str = Query("", max_length=200),
        refine: str = Query("", max_length=200),
    ) -> dict:
        results, elapsed_ms = engine.search_timed(q, limit=limit, title_contains=title, refine=refine)
        return {
            "query": q,
            "title_filter": title,
            "refine": refine,
            "elapsed_ms": round(elapsed_ms, 2),
            "results": results,
        }

    @app.get("/", response_class=HTMLResponse)
    def index(q: str = "", title: str = "", refine: str = "") -> str:
        results, elapsed_ms = (
            engine.search_timed(q, limit=10, title_contains=title, refine=refine)
            if q.strip()
            else ([], 0.0)
        )
        items = "".join(
            f'<article><h2><a href="{html.escape(item["url"])}" target="_blank">{html.escape(item["title"])}</a></h2>'
            f'<p>{html.escape(item["text"][:800])}</p><small>score: {item["score"]}</small></article>'
            for item in results
        )
        return f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本語Wikipedia高速検索</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;background:#f7f7f8;color:#222}}form{{display:flex;gap:8px}}input{{flex:1;font-size:1.1rem;padding:12px;border:1px solid #bbb;border-radius:8px}}button{{padding:0 20px;border:0;border-radius:8px;background:#2563eb;color:white;font-size:1rem}}article{{background:white;border-radius:10px;padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px #ddd}}h2{{margin:0 0 8px;font-size:1.2rem}}a{{color:#1558b0}}p{{line-height:1.7;margin:8px 0}}small{{color:#777}}.elapsed{{color:#666;font-size:.9rem}}</style>
<h1>日本語Wikipedia検索</h1>
<form><input name="q" value="{html.escape(q)}" placeholder="検索語を入力" autofocus><input name="refine" value="{html.escape(refine)}" placeholder="結果をさらに絞り込む（任意）"><input name="title" value="{html.escape(title)}" placeholder="タイトルで絞り込み（任意）"><button>検索</button></form>
{f'<p class="elapsed">検索時間: {elapsed_ms:.2f} ms（{len(results)}件表示）</p>' if q.strip() else ''}
{items or ('<p>検索語を入力してください。</p>' if not q else '<p>結果が見つかりませんでした。</p>')}
</html>"""

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--title", default="", help="記事タイトルに含める文字列")
    parser.add_argument("--refine", default="", help="結果をさらに絞り込む語句")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = SearchEngine(DB_PATH, use_model=not args.lexical_only)
    if args.query:
        results, elapsed_ms = engine.search_timed(
            args.query, title_contains=args.title, refine=args.refine
        )
        print(f"検索時間: {elapsed_ms:.2f} ms")
        for result in results:
            print(f"{result['title']}\t{result['score']:.4f}\t{result['url']}")
        return 0
    import uvicorn
    uvicorn.run(make_app(engine), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
