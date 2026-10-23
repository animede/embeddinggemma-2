# 日本語Wikipedia高速検索

日本語Wikipediaの最新版ダンプをSQLite FTS5に取り込み、`google/embeddinggemma-2`で候補を意味的に再ランキングするローカル検索アプリです。

`embeddinggemma-2`は検索用に、クエリへ `task: search result | query: ...`、文書へ `title: ... | text: ...` を使います。テキスト検索では画像・音声エンコーダーを読み込まず、GPUがある場合はbfloat16を使用します。

## セットアップ

プロジェクトディレクトリで実行します。

```bash
cd /path/to/embeddinggemma-2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

初回起動時にHugging Faceから`google/embeddinggemma-2`が自動取得されます。GPUが利用できる場合はGPUを使い、GPUがない場合はCPUで動作します。

GPUを使わずに起動する場合:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python app.py --port 8001
```

必要容量の目安は、モデル・依存関係を除いて、ダンプ約4.9GB、SQLite/FTS5索引約20GBです。

## ダウンロードと索引作成

最新版の日本語Wikipedia本文ダンプは約4.9GBです。画像ファイル本体や全編集履歴は含みません。

```bash
.venv/bin/python download_dump.py
.venv/bin/python build_index.py
```

ダウンロードは中断しても同じコマンドで再開できます。`build_index.py`は主名前空間のページだけを対象にし、検索・表示用テキストを各ページ最大12,000文字に抑えます。全件のSQLite/FTS索引はダンプより大きくなるため、空き容量に余裕を持たせてください。

すでに`data/jawiki.sqlite3`がある場合は、ダウンロードと索引作成を省略して起動できます。

## 起動

```bash
.venv/bin/python app.py --port 8001
```

ブラウザで <http://127.0.0.1:8001> を開きます。ポート8000が空いている場合は、`--port 8001`を`--port 8000`に変更できます。

### Web画面で検索

「検索語を入力」に検索語を入力して検索します。「結果をさらに絞り込む」欄に語句を入力すると、検索結果を追加条件で絞り込めます。「タイトルで絞り込み」では記事タイトルに含まれる文字列を指定できます。

### CLIで検索

検索欄（`--query`）に語句を追記して再検索できます。

```bash
.venv/bin/python app.py --query "東京の地下鉄"
```

さらに検索結果を絞り込む場合は、`--refine`を使います。`--query`の検索結果と`--refine`の語句の両方に該当するページを対象にします。

```bash
.venv/bin/python app.py --query "地下鉄" --refine "東京"
```

表示件数は`--limit`で指定できます（1〜50件）。

```bash
.venv/bin/python app.py --query "地下鉄" --refine "東京" --limit 20
```

記事タイトルで絞り込むこともできます。

```bash
.venv/bin/python app.py --query "地下鉄" --title "東京"
```

検索時間もCLIに表示されます。

### APIで検索

起動中のサーバーに対して、次のように検索できます。

```bash
curl 'http://127.0.0.1:8001/api/search?q=地下鉄&refine=東京&title=東京&limit=20'
```

レスポンスには`results`と`elapsed_ms`（検索時間、ミリ秒）が含まれます。Web画面にも表示件数の入力欄があります。

モデルを使わず、SQLite FTS5だけで確認する場合:

```bash
.venv/bin/python app.py --lexical-only --query "東京の地下鉄"
```

環境変数 `WIKI_DB`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`（128/256/512/768）で設定を変更できます。既定の256次元は検索速度と精度のバランスを取った設定です。
