# API設計書

## 概要

RAGシステムの外部インターフェース設計です。外部システムからの参照を前提とした検索API、管理API、認証APIを定義します。

## API基本方針

### RESTful API原則

- HTTPメソッドの適切な使用（GET, POST, PUT, DELETE）
- ステータスコードの統一
- JSON形式でのデータ交換
- バージョニング対応（/v1/...）

### 認証・認可（API基本方針）

- JWT Token + API Key による認証
- Role-based Access Control (RBAC)
- Rate Limiting による保護

## API エンドポイント一覧

### 検索API

| Method | Endpoint | 説明 |
|--------|----------|------|
| POST | `/v1/search` | ハイブリッド検索 |
| POST | `/v1/search/semantic` | セマンティック検索 |
| POST | `/v1/search/keyword` | キーワード検索 |
| GET | `/v1/search/suggestions` | 検索候補取得 |

### ドキュメント管理API

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/v1/documents` | ドキュメント一覧取得 |
| GET | `/v1/documents/{id}` | ドキュメント詳細取得 |
| POST | `/v1/documents` | ドキュメント登録 |
| PUT | `/v1/documents/{id}` | ドキュメント更新 |
| DELETE | `/v1/documents/{id}` | ドキュメント削除 |

### システム管理API

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/v1/health` | ヘルスチェック |
| GET | `/v1/metrics` | システムメトリクス |
| POST | `/v1/reindex` | インデックス再構築 |
| GET | `/v1/status` | システム状態取得 |

## 詳細API仕様

### 1. ハイブリッド検索API

#### ハイブリッド検索APIエンドポイント

```http
POST /v1/search
```

#### ハイブリッド検索APIリクエスト

```json
{
  "query": "JWT認証の実装方法",
  "filters": {
    "source_types": ["swagger", "confluence"],
    "languages": ["ja", "en"],
    "date_range": {
      "from": "2024-01-01",
      "to": "2024-12-31"
    },
    "tags": ["authentication", "security"]
  },
  "search_options": {
    "search_type": "hybrid", // "dense", "sparse", "hybrid"
    "max_results": 10,
    "min_score": 0.7,
    "include_metadata": true,
    "highlight": true
  },
  "ranking_options": {
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "rerank": true,
    "diversity": true
  }
}
```

#### ハイブリッド検索APIレスポンス

```json
{
  "query": "JWT認証の実装方法",
  "total_results": 25,
  "returned_results": 10,
  "search_time_ms": 145,
  "results": [
    {
      "document_id": "uuid-1234",
      "chunk_id": "uuid-5678",
      "score": 0.89,
      "chunk_type": "section",
      "title": "JWT認証の実装ガイド",
      "content": "JWT（JSON Web Token）を使用した認証システムの実装方法について...",
      "highlighted_content": "**JWT**（JSON Web Token）を使用した**認証**システムの**実装方法**について...",
      "source": {
        "type": "confluence",
        "url": "https://confluence.example.com/jwt-auth",
        "author": "developer@example.com",
        "last_updated": "2024-01-15T10:30:00Z"
      },
      "metadata": {
        "category": "authentication",
        "tags": ["JWT", "security", "implementation"],
        "language": "ja",
        "complexity": "medium"
      },
      "context": {
        "hierarchy_path": "1.2.3",
        "parent_sections": ["セキュリティ", "認証システム"],
        "related_chunks": ["uuid-9012", "uuid-3456"]
      }
    }
  ],
  "facets": {
    "source_types": {
      "confluence": 15,
      "swagger": 8,
      "git": 2
    },
    "categories": {
      "authentication": 12,
      "security": 8,
      "api_design": 5
    }
  },
  "suggestions": [
    "OAuth2認証の実装",
    "トークンリフレッシュの仕組み",
    "セキュリティベストプラクティス"
  ]
}
```

### 2. ドキュメント登録API

#### ドキュメント登録APIエンドポイント

```http
POST /v1/documents
```

#### ドキュメント登録APIリクエスト

```json
{
  "source_type": "confluence",
  "source_id": "page-12345",
  "title": "API認証仕様書",
  "content": "# API認証仕様書\n\n## 概要\n...",
  "file_type": "markdown",
  "language": "ja",
  "metadata": {
    "url": "https://confluence.example.com/page-12345",
    "author": "developer@example.com",
    "version": "2.1.0",
    "category": "api_specification",
    "tags": ["API", "authentication", "specification"]
  },
  "processing_options": {
    "chunk_strategy": "semantic",
    "max_chunk_size": 512,
    "enable_overlap": true,
    "extract_entities": true
  }
}
```

#### ドキュメント登録APIレスポンス

```json
{
  "document_id": "uuid-abcd",
  "status": "processing",
  "message": "ドキュメントの処理を開始しました",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "processing_details": {
    "estimated_chunks": 8,
    "estimated_tokens": 3456,
    "processing_queue_position": 2
  }
}
```

### 3. システム状態API

#### システム状態APIエンドポイント

```http
GET /v1/status
```

#### システム状態APIレスポンス

```json
{
  "system_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api_server": {
      "status": "healthy",
      "response_time_ms": 45,
      "active_connections": 23
    },
    "embedding_service": {
      "status": "healthy",
      "model_loaded": true,
      "gpu_memory_usage": "12.5GB/24GB",
      "processing_queue": 5
    },
    "vector_database": {
      "status": "healthy",
      "collections": {
        "dense_vectors": {
          "total_vectors": 125000,
          "index_status": "built"
        },
        "sparse_vectors": {
          "total_vectors": 125000,
          "index_status": "built"
        }
      }
    },
    "metadata_database": {
      "status": "healthy",
      "connection_pool": "8/20",
      "slow_queries": 0
    }
  },
  "statistics": {
    "total_documents": 5432,
    "total_chunks": 125000,
    "daily_searches": 890,
    "average_search_time_ms": 234
  }
}
```

### 4. システムメトリクスAPI

#### システムメトリクスAPIエンドポイント

```http
GET /v1/metrics
```

#### システムメトリクスAPIレスポンス

```json
{
  "performance_metrics": {
    "search_metrics": {
      "avg_response_time_ms": 234,
      "p95_response_time_ms": 456,
      "p99_response_time_ms": 678,
      "requests_per_second": 12.5,
      "error_rate_percent": 0.2
    },
    "embedding_metrics": {
      "documents_per_second": 850,
      "avg_processing_time_ms": 1234,
      "queue_depth": 5,
      "gpu_utilization_percent": 78
    }
  },
  "usage_metrics": {
    "daily_active_users": 45,
    "total_searches_today": 890,
    "popular_queries": [
      "API認証",
      "JWT実装",
      "セキュリティ設定"
    ]
  },
  "resource_metrics": {
    "cpu_usage_percent": 45,
    "memory_usage_percent": 67,
    "disk_usage_percent": 34,
    "network_io_mbps": 12.3
  }
}
```

## エラーハンドリング

### 標準エラーレスポンス

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "リクエストパラメータが不正です",
    "details": [
      {
        "field": "query",
        "message": "クエリは必須項目です"
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req-uuid-1234"
  }
}
```

### エラーコード一覧

| コード | HTTPステータス | 説明 |
|--------|----------------|------|
| `VALIDATION_ERROR` | 400 | リクエスト形式エラー |
| `AUTHENTICATION_ERROR` | 401 | 認証エラー |
| `AUTHORIZATION_ERROR` | 403 | 認可エラー |
| `NOT_FOUND` | 404 | リソースが見つからない |
| `RATE_LIMIT_EXCEEDED` | 429 | レート制限超過 |
| `INTERNAL_ERROR` | 500 | 内部サーバーエラー |
| `SERVICE_UNAVAILABLE` | 503 | サービス利用不可 |

## 認証・認可

### JWT Token 認証

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### API Key 認証

```http
X-API-Key: your-api-key-here
```

### ロール定義

```yaml
roles:
  viewer:
    permissions:
      - search:read
      - documents:read
  editor:
    permissions:
      - search:read
      - documents:read
      - documents:write
  admin:
    permissions:
      - "*"
```

## Rate Limiting

### 制限ルール

```yaml
rate_limits:
  search_api:
    requests_per_minute: 60
    burst_capacity: 10
  document_api:
    requests_per_minute: 30
    burst_capacity: 5
  management_api:
    requests_per_minute: 10
    burst_capacity: 2
```

### Rate Limit ヘッダー

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## API クライアント例

### Python SDK

```python
from rag_client import RAGClient

client = RAGClient(
    base_url="https://rag-api.example.com/v1",
    api_key="your-api-key",
    timeout=30
)

# 検索実行
results = client.search(
    query="JWT認証の実装方法",
    max_results=10,
    search_type="hybrid"
)

# ドキュメント登録
document_id = client.create_document(
    title="新しいAPI仕様",
    content="...",
    source_type="confluence"
)
```

### cURL 例

```bash
# 検索API
curl -X POST https://rag-api.example.com/v1/search \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "JWT認証の実装方法",
    "search_options": {
      "max_results": 10,
      "search_type": "hybrid"
    }
  }'

# ドキュメント登録API
curl -X POST https://rag-api.example.com/v1/documents \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "API認証仕様書",
    "content": "...",
    "source_type": "confluence"
  }'
```

## OpenAPI 仕様書

詳細なOpenAPI 3.0仕様書は別途 `openapi.yaml` ファイルとして提供し、Swagger UIでの確認とクライアント生成に使用します。

```yaml
openapi: 3.0.3
info:
  title: RAG System API
  description: システム開発情報RAGシステムのAPI
  version: 1.0.0
servers:
  - url: https://rag-api.example.com/v1
    description: Production server
paths:
  /search:
    post:
      summary: ハイブリッド検索
      # ... 詳細な仕様
```
