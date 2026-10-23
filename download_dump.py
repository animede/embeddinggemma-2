#!/usr/bin/env python3
"""Download the current Japanese Wikipedia article dump with resume support."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://dumps.wikimedia.org/jawiki/latest/"
    "jawiki-latest-pages-articles-multistream.xml.bz2"
)


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.stat().st_size if output.exists() else 0
    headers = {"User-Agent": "jawiki-embedding-search/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        if exc.code != 416:
            raise
        print(f"既存ファイルはすでに完全です: {output}")
        return

    status = getattr(response, "status", None)
    if existing and status != 206:
        print("サーバーが再開ダウンロードに対応していないため、最初から取得します。")
        existing = 0
        response.close()
        response = urlopen(Request(url, headers={"User-Agent": headers["User-Agent"]}), timeout=60)

    total = response.headers.get("Content-Length")
    total_bytes = (int(total) + existing) if total else None
    mode = "ab" if existing else "wb"
    downloaded = existing
    started = time.monotonic()
    last_print = 0.0

    with response, output.open(mode) as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_print >= 1.0:
                elapsed = max(now - started, 0.001)
                speed = downloaded / elapsed / 1024 / 1024
                if total_bytes:
                    pct = downloaded / total_bytes * 100
                    line = f"\r{downloaded / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB ({pct:5.1f}%) {speed:.1f} MiB/s"
                else:
                    line = f"\r{downloaded / 1024**3:.2f} GiB {speed:.1f} MiB/s"
                print(line, end="", flush=True)
                last_print = now

    print(f"\n完了: {output} ({downloaded / 1024**3:.2f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/jawiki-latest-pages-articles-multistream.xml.bz2"),
    )
    args = parser.parse_args()
    try:
        download(args.url, args.output)
    except KeyboardInterrupt:
        print("\n中断しました。次回は続きから再開できます。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

