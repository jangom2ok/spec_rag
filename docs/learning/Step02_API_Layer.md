# Step02: API層の設計と実装

## 🎯 この章の目標
FastAPI を基盤とした API 層の設計思想、エンドポイント実装、認証・認可システムを理解する

---

## 📋 概要

API層はシステムの入口として、外部システムからのリクエストを受け付け、適切なビジネスロジックに処理を委譲し、標準化されたレスポンスを返却する責務を担います。

### 🏗️ API層の構造

```
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

```python
class SearchMode(str, Enum):
    HYBRID = "hybrid"     # Dense + Sparse統合
    SEMANTIC = "semantic" # Dense重視
    KEYWORD = "keyword"   # Sparse重視

@router.post("/semantic", response_model=SearchResponse)
async def search_semantic(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """セマンティック検索（Dense Vector重視）"""
    request.search_mode = SearchMode.SEMANTIC
    return await search_documents(request, current_user, search_engine)
```

### 2. ドキュメント管理API (`app/api/documents.py`)

#### 🎯 責務
- ドキュメントのCRUD操作
- バッチ処理・同期処理の制御
- 処理状況の監視

#### 📊 エンドポイント一覧

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

#### 🎯 責務
- システム監視・メトリクス取得
- 再インデックス処理
- 管理者向け操作

#### 📊 エンドポイント一覧

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

#### 🎯 責務
- JWT認証・API Key認証
- ユーザー管理・ロール管理
- トークン管理（発行・失効）

#### 📊 エンドポイント一覧

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

#### 1. JWT Token認証
```python
async def get_current_user_jwt(
    authorization: str | None = Header(None)
) -> dict[str, Any]:
    """JWT認証"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    
    token = authorization.split(" ")[1]
    
    if is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    payload = verify_token(token)
    email = payload.get("sub")
    
    user = users_storage.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {**user, "email": email, "auth_type": "jwt"}
```

#### 2. API Key認証
```python
async def get_current_user_api_key(
    x_api_key: str | None = Header(None)
) -> dict[str, Any]:
    """API Key認証"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    api_key_info = validate_api_key(x_api_key)
    if not api_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "user_id": api_key_info["user_id"],
        "permissions": api_key_info["permissions"],
        "auth_type": "api_key",
    }
```

#### 3. 統合認証（フォールバック）
```python
async def get_current_user_or_api_key(
    authorization: str | None = Header(None), 
    x_api_key: str | None = Header(None)
) -> dict[str, Any]:
    """JWT認証またはAPI Key認証を試行"""
    # API Key認証を先に試行
    if x_api_key:
        api_key_info = validate_api_key(x_api_key)
        if api_key_info:
            return {
                "user_id": api_key_info["user_id"],
                "permissions": api_key_info["permissions"],
                "auth_type": "api_key",
            }
    
    # JWT認証を試行
    if authorization and authorization.startswith("Bearer "):
        # JWT処理...
        pass
    
    raise HTTPException(status_code=401, detail="Authentication required")
```

### 権限管理（RBAC）

#### 権限レベル
```python
class Permission(str, Enum):
    READ = "read"      # 検索・参照
    WRITE = "write"    # ドキュメント作成・更新
    DELETE = "delete"  # ドキュメント削除
    ADMIN = "admin"    # システム管理

# ロール定義例
ROLES = {
    "viewer": [Permission.READ],
    "editor": [Permission.READ, Permission.WRITE],
    "admin": [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN]
}
```

#### 権限チェック実装
```python
def check_permission(user: dict, required_permission: str) -> None:
    """権限チェック"""
    user_permissions = user.get("permissions", [])
    if required_permission not in user_permissions:
        raise HTTPException(
            status_code=403, 
            detail=f"{required_permission} permission required"
        )

@router.post("/documents/", status_code=201)
async def create_document(
    document: DocumentCreate,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """ドキュメント作成（write権限必須）"""
    check_permission(current_user, "write")
    # 処理続行...
```

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

```python
# ❌ 危険: 環境変数チェックが不十分
if os.getenv("TESTING"):  # 空文字でもTrueになる
    return {"user_id": "test", "permissions": ["admin"]}

# ✅ 安全: 厳密なチェック
if os.getenv("TESTING") == "true":
    return {"user_id": "test", "permissions": ["admin"]}
```

### 2. パスワードログ出力
**問題**: リクエストログにパスワードが記録される

```python
# ❌ 危険: パスワードがログに残る
logger.info(f"Login request: {request.dict()}")

# ✅ 安全: 機密情報を除外
safe_data = request.dict()
safe_data.pop("password", None)
logger.info(f"Login request: {safe_data}")
```

### 3. レート制限不備
**問題**: API Keyベースのレート制限の実装不備

```python
# ✅ API Keyごとのレート制限実装例
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

def get_api_key_limiter():
    def api_key_identifier(request: Request):
        api_key = request.headers.get("X-API-Key")
        return api_key or get_remote_address(request)
    return Limiter(key_func=api_key_identifier)

@router.post("/search/")
@limiter.limit("100/minute")  # API Keyごとに100回/分
async def search(request: Request, search_request: SearchRequest):
    # 検索処理...
    pass
```

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