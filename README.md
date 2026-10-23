# 日本語Wikipedia高速検索

日本語Wikipediaの最新版ダンプをSQLite FTS5に取り込み、`google/embeddinggemma-2`で候補を意味的に再ランキングするローカル検索アプリです。

`embeddinggemma-2`は検索用に、クエリへ `task: search result | query: ...`、文書へ `title: ... | text: ...` を使います。テキスト検索では画像・音声エンコーダーを読み込まず、GPUがある場合はbfloat16を使用します。

## セットアップ

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

初回のモデル起動時にHugging Faceからモデルが自動取得されます。GPUを使う場合は、環境に合ったCUDA対応PyTorchを先にインストールしてください。

## ダウンロードと索引作成

最新版の日本語Wikipedia本文ダンプは約4.9GBです。画像ファイル本体や全編集履歴は含みません。

```bash
python download_dump.py
python build_index.py
```

ダウンロードは中断しても同じコマンドで再開できます。`build_index.py`は主名前空間のページだけを対象にし、検索・表示用テキストを各ページ最大12,000文字に抑えます。全件のSQLite/FTS索引はダンプより大きくなるため、空き容量に余裕を持たせてください。

## 起動

```bash
python app.py --port 8001
```

ブラウザで <http://127.0.0.1:8001> を開きます。ポート8000が空いている場合は `--port 8001` を省略できます。CLIで検索する場合は次のようにします。

```bash
python app.py --query "東京の地下鉄"
```

記事タイトルで絞り込むこともできます。

```bash
python app.py --query "地下鉄" --title "東京"
```

検索結果をさらに語句で絞り込む場合は `--refine` を使います。

```bash
python app.py --query "地下鉄" --refine "東京"
```

APIでは `GET /api/search?q=地下鉄&refine=東京&title=東京&limit=10` を使います。

モデルを使わず、SQLite FTS5だけで確認する場合:

```bash
python app.py --lexical-only --query "東京の地下鉄"
```

環境変数 `WIKI_DB`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`（128/256/512/768）で設定を変更できます。既定の256次元は検索速度と精度のバランスを取った設定です。
