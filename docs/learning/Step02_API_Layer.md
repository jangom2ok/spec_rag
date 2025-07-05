# Step02: API層の設計と実装

## 🎯 この章の目標

FastAPI を基盤とした API 層の設計思想、エンドポイント実装、認証・認可システムを理解する

---

## 📋 概要

API層はシステムの入口として、外部システムからのリクエストを受け付け、適切なビジネスロジックに処理を委譲し、標準化されたレスポンスを返却する責務を担います。

### 🏗️ API層の構造

```text
app/api/
├── __init__.py
├── auth.py          # 認証・認可エンドポイント
├── documents.py     # ドキュメント管理API
├── health.py        # ヘルスチェックAPI
├── search.py        # 検索API (メイン機能)
└── system.py        # システム管理API
```

---

## 🔧 主要エンドポイント詳細

### 1. 検索API (`app/api/search.py`)

#### 🎯 責務

- ハイブリッド検索の実行
- セマンティック/キーワード検索の提供
- 検索候補・設定情報の提供

#### 📊 エンドポイント一覧

| エンドポイント | メソッド | 機能 | 認証 |
|---------------|----------|------|------|
| `/v1/search/` | POST | ハイブリッド検索 | 必須 |
| `/v1/search/semantic` | POST | セマンティック検索 | 必須 |
| `/v1/search/keyword` | POST | キーワード検索 | 必須 |
| `/v1/search/suggestions` | GET | 検索候補取得 | 必須 |
| `/v1/search/config` | GET | 検索設定取得 | 必須 |

#### 🔄 リクエスト/レスポンス設計

**ハイブリッド検索リクエスト例:**

```json
{
  "query": "FastAPI 認証システム 実装方法",
  "filters": {
    "source_types": ["git", "jira"],
    "languages": ["ja"],
    "date_range": {
      "from": "2024-01-01",
      "to": "2024-12-31"
    },
    "tags": ["authentication", "fastapi"]
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
    "diversity": false
  }
}
```

**レスポンス例:**

```json
{
  "query": "FastAPI 認証システム 実装方法",
  "total_results": 25,
  "returned_results": 10,
  "search_time_ms": 234.5,
  "results": [
    {
      "document_id": "doc_123",
      "chunk_id": "chunk_456",
      "score": 0.89,
      "title": "FastAPI認証システム実装ガイド",
      "content": "FastAPIでJWT認証を実装する方法...",
      "highlighted_content": "**FastAPI**でJWT**認証**を**実装**する方法...",
      "source": {
        "type": "confluence",
        "url": "https://wiki.example.com/pages/123",
        "author": "developer@example.com",
        "last_updated": "2024-01-15T10:30:00Z"
      },
      "metadata": {
        "tags": ["authentication", "fastapi", "jwt"],
        "category": "technical_guide",
        "difficulty": "intermediate"
      },
      "context": {
        "hierarchy_path": "開発ガイド > セキュリティ > 認証",
        "parent_sections": ["セキュリティ実装", "認証システム"],
        "related_chunks": ["chunk_457", "chunk_458"]
      }
    }
  ],
  "facets": {
    "source_type": {"git": 15, "jira": 10},
    "language": {"ja": 20, "en": 5},
    "tags": {"authentication": 18, "fastapi": 12, "jwt": 8}
  },
  "suggestions": [
    "FastAPI 認証システム 実装 JWT",
    "FastAPI セキュリティ設定",
    "認証システム FastAPI ベストプラクティス"
  ]
}
```

#### 🎛️ 検索モード実装

**実装ファイル**: `../../app/api/search.py`

本システムでは3つの検索モードを提供しており、用途に応じて最適な検索方式を選択できます：

**検索モードの種類**:
- **HYBRID**: Dense VectorとSparse Vectorを統合し、バランスの取れた検索結果を提供
- **SEMANTIC**: Dense Vector（意味的類似性）を重視した検索で、文脈を理解した結果を返却
- **KEYWORD**: Sparse Vector（キーワードマッチ）を重視した検索で、正確な用語の一致を優先

各検索モードは共通のインターフェースを持ち、`search_mode`パラメータの変更のみで切り替え可能です。内部では、重み付けパラメータが自動的に調整され、それぞれのモードに最適化された検索が実行されます。

### 2. ドキュメント管理API (`app/api/documents.py`)

#### 🎯 ドキュメント管理APIの責務

- ドキュメントのCRUD操作
- バッチ処理・同期処理の制御
- 処理状況の監視

#### 📊 ドキュメント管理APIのエンドポイント一覧

| エンドポイント | メソッド | 機能 | 権限 |
|---------------|----------|------|------|
| `/v1/documents/` | GET | ドキュメント一覧 | read |
| `/v1/documents/` | POST | ドキュメント作成 | write |
| `/v1/documents/{id}` | GET | ドキュメント取得 | read |
| `/v1/documents/{id}` | PUT | ドキュメント更新 | write |
| `/v1/documents/{id}` | DELETE | ドキュメント削除 | delete |
| `/v1/documents/process` | POST | バッチ処理実行 | write |
| `/v1/documents/process/sync` | POST | 同期処理実行 | write |

#### 🔄 ドキュメント処理リクエスト

```json
{
  "source_type": "confluence",
  "source_path": "/opt/documents",
  "file_patterns": ["*.md", "*.txt"],
  "batch_size": 50,
  "max_documents": 1000,
  "chunking_strategy": "semantic",
  "chunk_size": 1000,
  "overlap_size": 200,
  "extract_structure": true,
  "extract_entities": true,
  "extract_keywords": true,
  "max_concurrent_documents": 5
}
```

### 3. システム管理API (`app/api/system.py`)

#### 🎯 システム管理APIの責務

- システム監視・メトリクス取得
- 再インデックス処理
- 管理者向け操作

#### 📊 システム管理APIのエンドポイント一覧

| エンドポイント | メソッド | 機能 | 権限 |
|---------------|----------|------|------|
| `/v1/status` | GET | システム状態取得 | admin |
| `/v1/metrics` | GET | メトリクス取得 | admin |
| `/v1/reindex` | POST | 再インデックス実行 | admin |
| `/v1/reindex/{task_id}` | GET | タスク状況確認 | admin |

#### 🔄 システム状態レスポンス

```json
{
  "system_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api_server": {
      "status": "healthy",
      "response_time_ms": 45.0,
      "metadata": {
        "active_connections": 23,
        "uptime_seconds": 86400
      }
    },
    "embedding_service": {
      "status": "healthy",
      "metadata": {
        "model_loaded": true,
        "gpu_memory_usage": "12.5GB/24GB",
        "processing_queue": 5
      }
    },
    "vector_database": {
      "status": "healthy",
      "metadata": {
        "collections": {
          "dense_vectors": {"total_vectors": 125000, "index_status": "built"},
          "sparse_vectors": {"total_vectors": 125000, "index_status": "built"}
        }
      }
    }
  },
  "statistics": {
    "total_documents": 5432,
    "total_chunks": 125000,
    "daily_searches": 890,
    "average_search_time_ms": 234.0
  }
}
```

### 4. 認証API (`app/api/auth.py`)

#### 🎯 認証APIの責務

- JWT認証・API Key認証
- ユーザー管理・ロール管理
- トークン管理（発行・失効）

#### 📊 認証APIのエンドポイント一覧

| エンドポイント | メソッド | 機能 | 認証 |
|---------------|----------|------|------|
| `/auth/login` | POST | ログイン・トークン発行 | なし |
| `/auth/logout` | POST | ログアウト・トークン失効 | 必須 |
| `/auth/refresh` | POST | トークン更新 | 必須 |
| `/auth/users` | GET | ユーザー一覧 | admin |
| `/auth/users/{id}/roles` | PUT | ロール変更 | admin |

---

## 🔐 認証・認可システム

### 認証方式

**実装ファイル**: `../../app/core/auth.py`

本システムでは、柔軟な認証方式を提供するため、JWTトークンとAPIキーの両方に対応しています。

#### 1. JWT Token認証

**特徴**:
- **有効期限設定**: デフォルト24時間で自動失効
- **リフレッシュトークン**: 有効期限前に更新可能
- **ブラックリスト管理**: ログアウトやセキュリティ侵害時の即座失効
- **ペイロード情報**: ユーザーID、メール、権限情報を含む

**使用シーン**:
- Webアプリケーションのセッション管理
- モバイルアプリケーションの認証
- 短期間のアクセス制御

#### 2. API Key認証

**特徴**:
- **長期有効**: 明示的な失効まで有効
- **スコープ制限**: 特定のAPIのみアクセス可能
- **レート制限**: APIキーごとの利用回数制限
- **監査ログ**: 全てのアクセスを記録

**使用シーン**:
- システム間連携
- CI/CDパイプライン
- バッチ処理

#### 3. 統合認証（フォールバック）

システムはリクエストヘッダーを確認し、以下の優先順位で認証を試行します：

1. **X-API-Keyヘッダー**: APIキーによる認証
2. **Authorizationヘッダー**: BearerトークンによるJWT認証
3. どちらも存在しない場合は401エラーを返却

このアプローチにより、クライアントは状況に応じて最適な認証方式を選択できます。

### 権限管理（RBAC）

**実装ファイル**: `../../app/core/rbac.py`

ロールベースのアクセス制御（RBAC）を実装し、柔軟かつ安全な権限管理を実現しています。

#### 権限レベル

**4つの基本権限**:
- **READ**: 検索・参照のみ可能（デフォルト権限）
- **WRITE**: ドキュメントの作成・更新が可能
- **DELETE**: ドキュメントの削除が可能
- **ADMIN**: システム設定・ユーザー管理が可能

**ロール定義**:
- **viewer**: 閲覧専用ユーザー（READのみ）
- **editor**: コンテンツ編集者（READ + WRITE）
- **admin**: システム管理者（全権限）

#### 権限チェックの実装

権限チェックはデコレータパターンで実装され、エンドポイントごとに必要な権限を宣言的に指定できます。

**権限チェックの特徴**:
- **宣言的な権限指定**: `@require_permission("write")`のようにシンプルに指定
- **自動エラーハンドリング**: 権限不足時は403エラーを自動返却
- **ログ記録**: 全ての権限チェックを監査ログに記録
- **柔軟な権限組み合わせ**: 複数権限のAND/OR条件に対応

**権限の継承**:
上位の権限は下位の権限を含むため、ADMIN権限を持つユーザーは全ての操作が可能です。

---

## 🛡️ エラーハンドリング

### 構造化エラーレスポンス

全てのAPIエラーは統一フォーマットで返却されます：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "type": "validation_error",
    "details": [
      {
        "field": "query",
        "message": "This field is required"
      }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345678"
}
```

### エラーハンドラー実装 (`app/main.py`)

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors(),
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_code = "HTTP_ERROR"
    error_type = "http_error"

    # 認証エラーの場合
    if exc.status_code == 401:
        error_code = "AUTHENTICATION_ERROR"
        error_type = "authentication"
    elif exc.status_code == 403:
        error_code = "AUTHORIZATION_ERROR"
        error_type = "authorization"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": error_code,
                "message": str(exc.detail),
                "type": error_type,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )
```

---

## ⚡ パフォーマンス最適化

### 1. 依存性注入の最適化

```python
# ❌ 悪い例: 毎回新しいインスタンス
@router.post("/search/")
async def search(request: SearchRequest):
    search_engine = HybridSearchEngine()  # 毎回作成
    return await search_engine.search(request)

# ✅ 良い例: 依存性注入でインスタンス再利用
async def get_hybrid_search_engine() -> HybridSearchEngine:
    # 設定に基づいてインスタンス作成・キャッシュ
    return cached_search_engine

@router.post("/search/")
async def search(
    request: SearchRequest,
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine)
):
    return await search_engine.search(request)
```

### 2. レスポンスキャッシング

```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@router.get("/search/suggestions")
@cache(expire=300)  # 5分間キャッシュ
async def get_search_suggestions(
    q: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    # 検索候補生成（計算コストが高い）
    suggestions = await generate_suggestions(q, limit)
    return {"suggestions": suggestions, "query": q}
```

### 3. バックグラウンド処理

```python
from fastapi import BackgroundTasks

@router.post("/documents/process")
async def process_documents(
    config: ProcessingConfigRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """非同期でバックグラウンド処理"""
    # 即座にレスポンス返却
    task_id = str(uuid.uuid4())

    # バックグラウンドタスク追加
    background_tasks.add_task(
        process_documents_async,
        task_id, config
    )

    return {
        "success": True,
        "task_id": task_id,
        "message": "Processing started in background"
    }
```

---

## ❗ よくある落とし穴と対策

### 1. 認証バイパス

**問題**: テストモードでの認証スキップが本番環境に影響

**対策**:
- 環境変数の厳密なチェック（`== "true"`のように明示的に比較）
- テスト用の別設定ファイルを使用
- テストモードでも最小限の権限のみ付与
- CI/CDパイプラインでの環境変数検証

### 2. パスワードログ出力

**問題**: リクエストログにパスワードが記録される

**対策**:
- ログ出力前の機密情報フィルタリング
- ログミドルウェアでの自動マスキング
- 機密フィールドのリスト管理
- ログレビューの定期実施

### 3. レート制限不備

**問題**: API Keyベースのレート制限の実装不備

**対策**:
- APIキーごとのレート制限実装
- Redisを使用した分散環境でのカウンター管理
- 柔軟な制限ルール（スライディングウィンドウ、トークンバケット）
- レート制限超過時の適切なエラーメッセージ

**実装ファイル**: `../../app/core/rate_limiter.py`

レート制限の詳細な実装は上記ファイルで管理され、以下の機能を提供します：
- エンドポイント別の制限設定
- ユーザータイプ別の制限値
- バーストトラフィックの検出と対処

---

## 🎯 理解確認のための設問

### 基本理解

1. このシステムで使用されている2種類の認証方式の特徴と用途を説明してください
2. ハイブリッド検索、セマンティック検索、キーワード検索の違いを説明してください
3. 構造化エラーレスポンスに含まれる4つの必須フィールドを挙げてください

### API設計理解

1. `/v1/search/`エンドポイントの`filters`パラメータで指定可能な4種類のフィルター条件を説明してください
2. `search_options`の`highlight`機能が有効な場合のレスポンス形式の変化を説明してください
3. システム管理APIで管理者権限が必要な理由を3つ挙げてください

### 実装理解

1. 依存性注入を使用することで解決されるパフォーマンス問題を説明してください
2. バックグラウンド処理を使用する場面とその利点を2つ挙げてください
3. `get_current_user_or_api_key`でAPI Key認証を先に試行する理由を説明してください

### セキュリティ理解

1. JWT Token認証でトークンブラックリストをチェックする目的を説明してください
2. RBAC（Role-Based Access Control）の実装における権限チェックの仕組みを説明してください
3. APIキーベースのレート制限を実装する際の考慮点を2つ挙げてください

---

## 📚 次のステップ

API層の設計を理解できたら、次の学習段階に進んでください：

- **Step03**: ベクター検索エンジンの仕組み - BGE-M3とハイブリッド検索の内部実装
- **Step04**: 埋め込みサービスとBGE-M3 - ベクター生成の詳細プロセス
- **Step05**: データモデル設計 - PostgreSQL・Milvusのスキーマ詳細

API層は外部インターフェースとして重要な役割を果たします。次のステップでは、これらのAPIが内部でどのような処理を行っているかを詳しく学習します。
