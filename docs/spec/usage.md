# spec_rag システム利用ガイド

## 概要

spec_ragは、システム開発情報のためのRAG（Retrieval-Augmented Generation）システムです。BGE-M3モデルを使用したハイブリッド検索（Dense + Sparse + Multi-Vector）により、高精度な文書検索を提供します。

## システムの初期化

### 1. 前提条件

- Python 3.11以上
- Docker & Docker Compose
- 必要なストレージ容量: 最低100GB（ベクトルデータベース用）

### 2. 環境構築

#### 2.1 リポジトリのクローン

```bash
git clone <repository-url>
cd spec_rag
```

#### 2.2 仮想環境の作成と依存関係のインストール

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -e ".[dev]"
```

#### 2.3 環境変数の設定

`.env`ファイルを作成し、以下の設定を記述：

```bash
# データベース設定
DATABASE_URL=postgresql://user:password@localhost:5432/spec_rag
REDIS_URL=redis://localhost:6379/0

# ApertureDB設定
APERTUREDB_HOST=localhost
APERTUREDB_PORT=55555

# アプリケーション設定
ENVIRONMENT=development
LOG_LEVEL=INFO

# 認証設定
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

JWT_SECRET_KEYを生成するコマンド:

```bash
python -c "import secrets; print('JWT_SECRET_KEY =', repr(secrets.token_urlsafe(32)))"
```

#### 2.4 Docker環境の起動

```bash
# ApertureDB、PostgreSQL、Redisを起動
docker-compose up -d
```

#### 2.5 データベースの初期化

##### 初回セットアップ

```bash
# PostgreSQLドライバのインストール（必須）
pip install psycopg2-binary

# Alembicの初期化（初回のみ）
alembic init alembic
```

初回実行時は、以下の設定が必要です：

1. **alembic.ini の編集**
   ```ini
   # sqlalchemy.url の行をコメントアウト
   # sqlalchemy.url = driver://user:pass@localhost/dbname
   ```

2. **alembic/env.py の編集**
   ```python
   # ファイルの先頭に追加
   import os
   import sys
   from pathlib import Path

   # プロジェクトルートをPythonパスに追加
   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

   # アプリケーションの設定とモデルをインポート
   from app.core.config import settings
   from app.models.database import Base, Document, DocumentChunk

   # データベースURLを環境変数から設定
   config.set_main_option("sqlalchemy.url", settings.database_url)

   # target_metadataを設定
   target_metadata = Base.metadata
   ```

3. **初期マイグレーションの作成**
   ```bash
   # 自動的にモデルからマイグレーションを生成
   alembic revision --autogenerate -m "Initial migration"
   ```

4. **マイグレーションの適用**
   ```bash
   # データベースにテーブルを作成
   alembic upgrade head
   ```

##### 2回目以降の実行

既にAlembicが初期化されている場合は、以下のコマンドのみ実行：

```bash
# 最新のマイグレーションを適用
alembic upgrade head
```

##### マイグレーションの確認

```bash
# 現在のマイグレーション状態を確認
alembic current

# 適用可能なマイグレーションの一覧
alembic history
```

##### トラブルシューティング

**エラー: `No module named 'psycopg2'`**

```bash
pip install psycopg2-binary
```

**エラー: `FAILED: No 'script_location' key found in configuration`**

- プロジェクトルートディレクトリで実行していることを確認
- `alembic init alembic` を実行してAlembicを初期化

**エラー: `name 'app' is not defined`**

- マイグレーションファイルに `import app.models.database` を追加

**エラー: `data type json has no default operator class for access method "gin"`**

- PostgreSQLのJSON型はGINインデックスをサポートしていません
- JSONB型を使用するか、GINインデックスを削除してください

### 3. アプリケーションの起動

```bash
# FastAPIアプリケーションの起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

起動後、以下のURLでアクセス可能：

- API: http://localhost:8000
- API Documentation (Swagger UI): http://localhost:8000/docs
- Alternative API Documentation (ReDoc): http://localhost:8000/redoc

## APIの使い方

### 認証

spec_ragは2種類の認証方式をサポートしています：

#### JWT Token認証

##### 1. ユーザー登録

```bash
# 新しいユーザーを登録
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "your-password",
    "full_name": "Test User"
  }'
```

##### 2. ログイン（OAuth2標準フォーマット）

```bash
# ログインしてトークンを取得
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=your-password"

# レスポンス例
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

##### 3. 認証が必要なAPIへのアクセス

```bash
# トークンを使用してAPIにアクセス
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  http://localhost:8000/v1/search
```

#### API Key認証

```bash
# API Keyを使用してアクセス
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:8000/v1/search
```

### 主要APIエンドポイント

#### 1. ハイブリッド検索

システムの中心機能である、BGE-M3を使用したハイブリッド検索です。

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "JWT認証の実装方法",
    "filters": {
      "source_types": ["git", "confluence"],
      "languages": ["ja"],
      "date_range": {
        "from": "2024-01-01",
        "to": "2024-12-31"
      }
    },
    "search_options": {
      "search_type": "hybrid",
      "max_results": 10,
      "min_score": 0.5,
      "include_metadata": true,
      "highlight": true
    },
    "ranking_options": {
      "dense_weight": 0.7,
      "sparse_weight": 0.3,
      "rerank": true,
      "diversity": true
    }
  }'
```

#### 2. ドキュメント登録

新しいドキュメントをシステムに登録します。

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "git",
    "source_id": "unique-doc-id",
    "title": "JWT認証実装ガイド",
    "content": "# JWT認証の実装\n\n## 概要\nJWT（JSON Web Token）は...",
    "file_type": "markdown",
    "language": "ja",
    "metadata": {
      "url": "https://github.com/example/repo/blob/main/auth.md",
      "author": "developer@example.com",
      "tags": ["authentication", "jwt", "security"]
    },
    "processing_options": {
      "chunk_strategy": "semantic",
      "max_chunk_size": 512,
      "enable_overlap": true
    }
  }'
```

#### 3. ドキュメント一覧取得

```bash
curl -X GET "http://localhost:8000/v1/documents?page=1&per_page=20" \
  -H "Authorization: Bearer $TOKEN"
```

#### 4. 検索候補の取得

ユーザーの入力に基づいて検索候補を提供します。

```bash
curl -X GET "http://localhost:8000/v1/search/suggestions?q=JWT" \
  -H "Authorization: Bearer $TOKEN"
```

#### 5. システム状態の確認

```bash
curl -X GET http://localhost:8000/v1/status \
  -H "Authorization: Bearer $TOKEN"
```

### 検索タイプ

spec_ragは3つの検索タイプをサポートしています：

1. **hybrid** (推奨): Dense、Sparse、Multi-Vectorを組み合わせた最も精度の高い検索
2. **dense**: 意味的類似性に基づく検索（セマンティック検索）
3. **sparse**: キーワードベースの検索（従来型検索）

### フィルターオプション

検索結果を絞り込むための様々なフィルターを利用できます：

- **source_types**: ドキュメントのソースタイプ（git、confluence、jira等）
- **languages**: ドキュメントの言語（ja、en等）
- **tags**: ドキュメントに付与されたタグ
- **date_range**: ドキュメントの更新日時範囲
- **document_types**: ドキュメントの種類（guide、api、specification等）

### ランキングオプション

検索結果の品質を向上させるオプション：

- **dense_weight / sparse_weight**: ハイブリッド検索時の重み調整（合計が1.0になるよう正規化）
- **rerank**: 再ランキング機能の有効化（CrossEncoderを使用）
- **diversity**: 結果の多様性を確保（MMRアルゴリズムを使用）

## 高度な使用方法

### バッチ処理

大量のドキュメントを効率的に処理する場合：

```python
import asyncio
from app.services.document_service import DocumentService

async def batch_import_documents(documents):
    service = DocumentService()

    # バッチサイズ100でドキュメントを処理
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        await service.bulk_create_documents(batch)
        print(f"Processed {i+batch_size} documents")

# 使用例
documents = [
    {
        "title": "Document 1",
        "content": "...",
        "source_type": "git"
    },
    # ... 更多のドキュメント
]

asyncio.run(batch_import_documents(documents))
```

### カスタム処理オプション

ドキュメント登録時に、用途に応じた処理オプションを指定できます：

```json
{
  "processing_options": {
    "chunk_strategy": "semantic",     // "fixed_size", "semantic", "hierarchical"
    "max_chunk_size": 512,           // トークン数
    "overlap_size": 50,              // チャンク間のオーバーラップ
    "enable_overlap": true,
    "extract_entities": true,        // エンティティ抽出
    "generate_summary": true,        // 要約生成
    "detect_language": true          // 言語自動検出
  }
}
```

## パフォーマンスチューニング

### 検索パフォーマンスの最適化

1. **適切な検索タイプの選択**
   - 精度重視: `hybrid`
   - 速度重視: `sparse`
   - セマンティック検索: `dense`

2. **結果数の制限**
   - `max_results`を必要最小限に設定
   - ページネーションの活用

3. **キャッシュの活用**
   - 頻繁に検索されるクエリは自動的にキャッシュされます
   - キャッシュの有効期限: 15分（デフォルト）

### システムメトリクスの監視

```bash
# メトリクスの取得
curl -X GET http://localhost:8000/v1/metrics \
  -H "Authorization: Bearer $TOKEN"
```

主要なメトリクス：
- 平均検索応答時間
- 95パーセンタイル応答時間
- 1秒あたりのリクエスト数
- エラー率
- GPU使用率（埋め込み生成時）

## トラブルシューティング

### よくある問題と解決方法

1. **検索結果が返ってこない**
   - ドキュメントが正しくインデックスされているか確認
   - フィルター条件が厳しすぎないか確認
   - `min_score`の値を下げてみる

2. **検索が遅い**
   - `search_type`を`sparse`に変更してパフォーマンスを確認
   - `max_results`を減らす
   - システムリソース（CPU、メモリ、GPU）を確認

3. **認証エラー**
   - トークンの有効期限を確認
   - API Keyが正しく設定されているか確認
   - ユーザーの権限を確認

### ログの確認

```bash
# アプリケーションログ
tail -f logs/app.log

# エラーログ
tail -f logs/error.log

# アクセスログ
tail -f logs/access.log
```

## セキュリティベストプラクティス

1. **認証情報の管理**
   - 環境変数で機密情報を管理
   - 定期的なトークンのローテーション
   - 強力なパスワードポリシーの適用

2. **APIアクセス制御**
   - Rate Limitingの適切な設定
   - IPホワイトリストの活用
   - CORSの適切な設定

3. **データ保護**
   - HTTPSの使用（本番環境）
   - データベースの暗号化
   - バックアップの定期実行

## まとめ

spec_ragは、高度な検索機能を提供する柔軟なRAGシステムです。本ガイドに従って初期設定を行い、APIを通じて強力な検索機能を活用してください。

詳細な技術仕様については、以下のドキュメントも参照してください：
- [API設計書](../develop/detailed_design/APIDesign.md)
- [システムアーキテクチャ](../develop/detailed_design/SystemArchitecture.md)
- [データモデル設計](../develop/detailed_design/DataModel.md)
EOF < /dev/null
