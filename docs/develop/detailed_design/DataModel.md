# データモデル設計書

## 概要

RAGシステムにおけるデータモデルの詳細設計です。ドキュメントの粒度、構造化、メタデータ管理、およびベクトル化戦略を定義します。

## ドキュメント粒度戦略

### 基本方針

RAG.mdで述べられているように、「ドキュメント粒度、1つの1つの情報の抽象度、話題、カテゴリ、情報をどう分割して保存するか」を明確にします。

### 粒度レベル定義

| レベル | 名称 | 説明 | 例 | チャンクサイズ |
|--------|------|------|----|-----------|
| 1 | **Document** | 1つの完全なドキュメント | API仕様書、設計書 | 全文 (～8k tokens) |
| 2 | **Section** | 章・セクション単位 | API Endpointの説明 | 500-2000 tokens |
| 3 | **Paragraph** | 段落・項目単位 | 特定の機能説明 | 100-500 tokens |
| 4 | **Term** | 用語・定義単位 | 用語集エントリ | 10-100 tokens |

## データスキーマ設計

### 1. Documents テーブル (PostgreSQL)

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL, -- 'git', 'swagger', 'sheets'
    source_id VARCHAR(255) NOT NULL, -- 元システムでの識別子
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL, -- SHA-256
    file_type VARCHAR(50), -- 'markdown', 'json', 'yaml', 'pdf'
    language VARCHAR(10) DEFAULT 'ja', -- 'ja', 'en', 'mixed'
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'archived', 'deleted'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(source_type, source_id)
);

CREATE INDEX idx_documents_source ON documents(source_type, source_id);
CREATE INDEX idx_documents_updated ON documents(updated_at);
CREATE INDEX idx_documents_hash ON documents(content_hash);
```

### 2. Document Chunks テーブル

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(20) NOT NULL, -- 'section', 'paragraph', 'term'
    title TEXT,
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    token_count INTEGER,
    hierarchy_path TEXT, -- '1.2.3' などの階層表現
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_type ON document_chunks(chunk_type);
CREATE INDEX idx_chunks_metadata ON document_chunks USING GIN(metadata);
```

### 3. Vector Descriptor Sets (ApertureDB)

#### Dense Vector Descriptor Set

```python
# Descriptor Set Schema
dense_descriptor_set_schema = {
    "name": "document_vectors_dense",
    "fields": [
        {
            "name": "id",
            "type": "VARCHAR",
            "max_length": 36,
            "is_primary": True
        },
        {
            "name": "document_id",
            "type": "VARCHAR",
            "max_length": 36
        },
        {
            "name": "chunk_id",
            "type": "VARCHAR",
            "max_length": 36
        },
        {
            "name": "vector",
            "type": "FLOAT_VECTOR",
            "dim": 1024
        },
        {
            "name": "chunk_type",
            "type": "VARCHAR",
            "max_length": 20
        },
        {
            "name": "source_type",
            "type": "VARCHAR",
            "max_length": 50
        },
        {
            "name": "language",
            "type": "VARCHAR",
            "max_length": 10
        },
        {
            "name": "created_at",
            "type": "INT64"
        }
    ]
}
```

#### Sparse Vector Descriptor Set

```python
# Sparse Vector用のディスクリプタセット
sparse_descriptor_set_schema = {
    "name": "document_vectors_sparse",
    "fields": [
        {
            "name": "id",
            "type": "VARCHAR",
            "max_length": 36,
            "is_primary": True
        },
        {
            "name": "document_id",
            "type": "VARCHAR",
            "max_length": 36
        },
        {
            "name": "chunk_id",
            "type": "VARCHAR",
            "max_length": 36
        },
        {
            "name": "sparse_vector",
            "type": "SPARSE_FLOAT_VECTOR"
        },
        {
            "name": "vocabulary_size",
            "type": "INT32"
        }
    ]
}
```

## メタデータスキーマ

### Document メタデータ

```json
{
    "source_metadata": {
        "url": "https://example.com/doc",
        "author": "user@example.com",
        "version": "1.2.0",
        "last_modified": "2024-01-01T00:00:00Z"
    },
    "processing_metadata": {
        "embedding_model": "BAAI/bge-m3",
        "chunk_strategy": "semantic",
        "total_chunks": 15,
        "average_chunk_size": 384
    },
    "domain_metadata": {
        "category": "api_specification",
        "tags": ["REST", "authentication", "v2"],
        "complexity": "medium",
        "importance": "high"
    }
}
```

### Chunk メタデータ

```json
{
    "position_metadata": {
        "start_line": 45,
        "end_line": 78,
        "start_char": 1234,
        "end_char": 2567
    },
    "content_metadata": {
        "headings": ["API仕様", "認証", "JWT"],
        "code_blocks": 2,
        "tables": 1,
        "links": ["https://example.com"]
    },
    "semantic_metadata": {
        "topics": ["authentication", "security", "API"],
        "entities": ["JWT", "OAuth2", "Bearer Token"],
        "keywords": ["認証", "トークン", "セキュリティ"]
    }
}
```

## ドキュメントタイプ別の処理戦略

### 1. API仕様書 (Swagger/OpenAPI)

```yaml
処理方針:
  - エンドポイント単位でセクション分割
  - パラメータ、レスポンスを個別のチャンクとして保存
  - スキーマ定義は用語として扱う

粒度:
  - Document: 全体のAPI仕様
  - Section: 各エンドポイント
  - Paragraph: パラメータ説明、レスポンス例
  - Term: データモデル定義
```

### 2. 設計書・仕様書 (Markdown)

```yaml
処理方針:
  - ヘッダー階層に基づいたセクション分割
  - 図表は別チャンクとして保存
  - 用語定義を自動抽出

粒度:
  - Document: 全体の設計書
  - Section: 章・セクション (H1, H2)
  - Paragraph: 小項目 (H3, H4)
  - Term: 定義、用語集
```

### 3. コードドキュメント (Git Repository)

```yaml
処理方針:
  - ファイル単位、関数単位で分割
  - コメント・docstringを重視
  - READMEとコードを関連付け

粒度:
  - Document: README、設計ファイル
  - Section: クラス、モジュール
  - Paragraph: 関数、メソッド
  - Term: 定数、設定値
```

### 4. 用語集・FAQ (Google Sheets)

```yaml
処理方針:
  - 1行1エントリで Term レベル
  - カテゴリ別にグループ化
  - 質問・回答をペアで保存

粒度:
  - Section: カテゴリ
  - Term: 個別の用語・Q&A
```

## チャンク化アルゴリズム

### セマンティック分割

```python
def semantic_chunking(text: str, max_tokens: int = 512) -> List[Chunk]:
    """
    意味的な境界を考慮したチャンク分割
    """
    # 1. 構造的境界の検出 (ヘッダー、リスト、改行)
    structural_boundaries = detect_structural_boundaries(text)

    # 2. 意味的境界の検出 (文章の類似度)
    semantic_boundaries = detect_semantic_boundaries(text)

    # 3. 境界の統合とチャンク生成
    boundaries = merge_boundaries(structural_boundaries, semantic_boundaries)
    chunks = create_chunks(text, boundaries, max_tokens)

    return chunks
```

### オーバーラップ戦略

```python
OVERLAP_CONFIG = {
    "section": {
        "overlap_tokens": 50,  # 前後50トークンの重複
        "boundary_preservation": True
    },
    "paragraph": {
        "overlap_tokens": 20,
        "boundary_preservation": False
    },
    "term": {
        "overlap_tokens": 0,  # 用語は重複なし
        "boundary_preservation": True
    }
}
```

## データ品質管理

### 重複検出

```sql
-- コンテンツハッシュによる重複検出
SELECT content_hash, COUNT(*) as duplicate_count
FROM documents
GROUP BY content_hash
HAVING COUNT(*) > 1;
```

### データ検証ルール

```yaml
validation_rules:
  documents:
    - title: NOT NULL, length > 0
    - content: NOT NULL, length > 10
    - content_hash: SHA-256 format
    - language: IN ('ja', 'en', 'mixed')

  chunks:
    - content_length: > 0
    - token_count: > 0, < 8192
    - chunk_index: >= 0
```

### 更新追跡

```sql
-- 更新履歴テーブル
CREATE TABLE document_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    change_type VARCHAR(20) NOT NULL, -- 'created', 'updated', 'deleted'
    old_content_hash VARCHAR(64),
    new_content_hash VARCHAR(64),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    change_summary TEXT
);
```

## パフォーマンス最適化

### インデックス戦略

```sql
-- PostgreSQL インデックス
CREATE INDEX CONCURRENTLY idx_chunks_content_gin
ON document_chunks USING GIN(to_tsvector('japanese', content));

CREATE INDEX CONCURRENTLY idx_documents_title_gin
ON documents USING GIN(to_tsvector('japanese', title));
```

### ApertureDB パーティション戦略

```python
# ソースタイプ別パーティション
PARTITIONS = [
    "git_documents",
    "swagger_documents",
    "sheets_documents"
]

# 日付別パーティション（アーカイブ用）
DATE_PARTITIONS = [
    "documents_2024_q1",
    "documents_2024_q2",
    "documents_2024_q3",
    "documents_2024_q4"
]
```
