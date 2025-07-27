# 外部の依存関係によるテスト失敗の修正計画

## 概要

この文書では、spec_rag プロジェクトにおける外部の依存関係によるテスト失敗の修正計画を説明します。

## テスト失敗のカテゴリ

### 1. Celery/Redis テスト (10 テスト)

**ファイル**: `test_embedding_tasks.py`, `test_redis_integration.py`

**問題点**:

- 実際の Redis インスタンスへの接続を試行するテスト
- ブロッカーなしで Celery タスクを実行しようとする
- タスクの状態チェックが失敗する

**解決策**:

- `mock_celery_app` と `mock_redis_client` フィクスチャを使用
- タスクの送信と結果の取得をモック化
- メモリ内タスク追跡を使用

### 2. 外部 API テスト (11 テスト)

**ファイル**: `test_external_source_integration.py`

**問題点**:

- 存在しない Confluence/JIRA インスタンスへの HTTP リクエスト
- API エンドポイントからの 404/400 エラー
- 認証失敗

**解決策**:

- `mock_httpx_client` フィクスチャを使用
- 現実的なデータでAPIレスポンスをモック化
- モック化されたエラーレスポンスでエラー処理をテスト

### 3. データベース接続テスト(8 テスト)

**ファイル**: `test_production_database.py`

**問題点**:

- PostgreSQL接続試行の失敗
- ApertureDBクライアントの初期化エラー
- 接続プール作成の失敗

**解決策**:

- `mock_asyncpg_pool` と `mock_aperturedb_client` フィクスチャを使用する
- ヘルスチェック応答をモック化する
- エラー処理テスト用に接続失敗をシミュレートする

### 4. GPU/ハードウェアテスト (2 テスト)

**ファイル**: `test_embedding_optimization.py`

**問題**:

- テスト環境でCUDAが利用不可
- GPUメモリチェックの失敗

**解決策**:

- `mock_cuda_available` と `mock_gpu_memory` フィクスチャを使用
- ハードウェア機能をモック化
- 実際のGPUなしで最適化ロジックをテスト

### 5. NLPライブラリテスト (4件)

**ファイル**: `test_metadata_extractor.py`

**問題点**:

- spaCy モデルの読み込み失敗
- エンティティ抽出に実際の NLP モデルが必要

**解決策**:

- `mock_spacy_model` フィクスチャを使用
- エンティティとキーワード抽出の結果をモック化
- モデル依存なしで処理ロジックをテスト

### 6. ベクターデータベーステスト (1 テスト)

**ファイル**: `test_error_handling.py`

**問題点**:

- エラー処理テストにはApertureDB接続が必要

**解決策**:

- `mock_aperturedb_client` フィクスチャを使用
- さまざまなエラー条件をシミュレート
- エラー回復ロジックをテスト

## 実装戦略

### フェーズ1: インフラストラクチャのセットアップ

1. ✅ `conftest_extended.py` にすべてのモックフィクスチャを作成
2. pytest を拡張フィクスチャ使用するように設定
3. 外部の依存関係テスト用のテストマーカーを追加

### フェーズ2: テストファイルの修正 (優先順位順)

1. **test_redis_integration.py** - 非同期タスク処理に不可欠
2. **test_embedding_tasks.py** - コア機能
3. **test_external_source_integration.py** - 外部データ取り込み
4. **test_production_database.py** - データベースのセットアップと健全性
5. **test_metadata_extractor.py** - ドキュメント処理
6. **test_embedding_optimization.py** - 性能最適化
7. **test_error_handling.py** - エラー回復

### フェーズ3: 検証

1. 各テストファイルをモックを使用して個別に実行
2. 外部の接続が試行されないことを確認
3. テストカバレッジが適切であることを確認
4. フルテストスイートを実行

## テストマーカー

これらのマーカーを追加してテストを分類します:

```python
# pyproject.toml 内で
[tool.pytest.ini_options]
markers = [
    「external: 外部サービスが必要なテストをマーク」,
    「unit: 外部依存関係なしで実行すべきユニットテストをマーク」,
    「integration: 統合テストをマーク」,
    「slow: 実行が遅いテストをマーク」,
]
```

## 環境変数

テストはこれらの環境変数を尊重する必要があります:

```bash
TESTING=true                    # テストモードを有効化
DISABLE_EXTERNAL_APIS=true     # 外部API呼び出しを無効化
USE_MOCK_EMBEDDINGS=true       # モック埋め込みを使用
MOCK_HARDWARE=true             # GPU/ハードウェアチェックをモック化
```

## テストの実行

```bash
# モックを使用してすべてのテストを実行
TESTING=true pytest

# ユニットテストのみを実行
pytest -m 「unit」

# 外部依存関係を除外してテストを実行
pytest -m 「not external」

# 特定のテストファイルを詳細出力で実行
pytest tests/test_redis_integration.py -v

# カバレッジ付きで実行
pytest --cov=app --cov-report=html
```

## 成功基準

1. 38件の失敗テストがモックで通過する
2. テスト中に外部接続が試行されない
3. テスト実行時間が大幅に短縮される
4. 外部サービスなしでCI/CD環境でテストを実行できる
5. テストカバレッジが維持または向上する

## 注意事項

- コードベースには既に一部のサービスでモック実装が存在する（例：`embedding_tasks.py`）
- 既存のモッククラスを可能な限り活用する
- モックが現実的なものとなるよう、実際の課題を捕捉できるようにする
- 外部サービスが本当に必要なテストには統合テストマーカーを追加する

DeepL.com（無料版）で翻訳しました。
