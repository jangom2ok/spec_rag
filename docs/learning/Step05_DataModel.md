# Step05: データモデル設計とスキーマ詳細

## 🎯 この章の目標
PostgreSQL・Milvusでのデータモデル設計、スキーマ詳細、インデックス戦略、データ整合性の仕組みを理解する

---

## 📋 概要

RAGシステムでは、構造化データ（メタデータ）と非構造化データ（ベクター）を効率的に管理するため、PostgreSQLとMilvusを使い分けています。適切なスキーマ設計により、高速検索と拡張性を両立します。

### 🏗️ データベース構成

```
データ保存戦略
├── PostgreSQL        # 構造化データ・メタデータ
│   ├── documents     # ドキュメント基本情報
│   ├── chunks        # チャンク詳細
│   ├── sources       # ソース管理
│   └── users         # ユーザー・認証
├── Milvus            # ベクターデータ
│   ├── dense_collection    # Dense vectors
│   ├── sparse_collection   # Sparse vectors
│   └── multi_collection    # Multi-vectors
└── Redis             # キャッシュ・セッション
    ├── search_cache  # 検索結果キャッシュ
    ├── embedding_cache # 埋め込みキャッシュ
    └── session_store # ユーザーセッション
```

---

## 🗃️ PostgreSQL スキーマ設計

### 1. ドキュメント管理テーブル

#### `documents` テーブル
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(1000) NOT NULL,
    content TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    source_url TEXT,
    
    -- メタデータ
    author VARCHAR(255),
    language CHAR(2) DEFAULT 'ja',
    category VARCHAR(100),
    tags TEXT[], -- PostgreSQL配列
    
    -- 統計情報
    word_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    
    -- 処理状況
    processing_status VARCHAR(20) DEFAULT 'pending',
    indexing_status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    
    -- タイムスタンプ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    indexed_at TIMESTAMP WITH TIME ZONE,
    
    -- 制約
    CONSTRAINT valid_processing_status 
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    CONSTRAINT valid_indexing_status 
        CHECK (indexing_status IN ('pending', 'indexing', 'completed', 'failed')),
    CONSTRAINT valid_language 
        CHECK (language ~ '^[a-z]{2}$'),
    CONSTRAINT positive_counts 
        CHECK (word_count >= 0 AND char_count >= 0 AND chunk_count >= 0)
);

-- インデックス
CREATE INDEX idx_documents_source ON documents (source_type, source_id);
CREATE INDEX idx_documents_status ON documents (processing_status, indexing_status);
CREATE INDEX idx_documents_created ON documents (created_at DESC);
CREATE INDEX idx_documents_language ON documents (language);
CREATE INDEX idx_documents_category ON documents (category);
CREATE INDEX idx_documents_tags ON documents USING GIN (tags);
CREATE UNIQUE INDEX idx_documents_source_unique ON documents (source_type, source_id);

-- 全文検索インデックス
CREATE INDEX idx_documents_fulltext ON documents 
    USING GIN (to_tsvector('japanese', title || ' ' || content));
```

#### `document_chunks` テーブル
```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- チャンク内容
    content TEXT NOT NULL,
    chunk_type VARCHAR(50) DEFAULT 'text',
    position INTEGER NOT NULL,
    
    -- 構造情報
    section_title VARCHAR(500),
    hierarchy_level INTEGER DEFAULT 1,
    parent_chunk_id UUID REFERENCES document_chunks(id),
    
    -- サイズ情報
    token_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    
    -- ベクター関連
    dense_vector_id VARCHAR(255), -- Milvus内のID
    sparse_vector_id VARCHAR(255),
    multi_vector_id VARCHAR(255),
    
    -- 処理状況
    embedding_status VARCHAR(20) DEFAULT 'pending',
    embedding_error TEXT,
    
    -- スコア情報（検索時に更新）
    last_search_score FLOAT,
    search_count INTEGER DEFAULT 0,
    
    -- タイムスタンプ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedded_at TIMESTAMP WITH TIME ZONE,
    
    -- 制約
    CONSTRAINT valid_embedding_status 
        CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed')),
    CONSTRAINT positive_position CHECK (position >= 0),
    CONSTRAINT positive_counts 
        CHECK (token_count >= 0 AND char_count >= 0 AND hierarchy_level >= 1)
);

-- インデックス
CREATE INDEX idx_chunks_document ON document_chunks (document_id, position);
CREATE INDEX idx_chunks_embedding_status ON document_chunks (embedding_status);
CREATE INDEX idx_chunks_vector_ids ON document_chunks (dense_vector_id, sparse_vector_id);
CREATE INDEX idx_chunks_hierarchy ON document_chunks (hierarchy_level, section_title);
CREATE INDEX idx_chunks_parent ON document_chunks (parent_chunk_id);

-- 全文検索
CREATE INDEX idx_chunks_fulltext ON document_chunks 
    USING GIN (to_tsvector('japanese', content));
```

### 2. ソース管理テーブル

#### `sources` テーブル
```sql
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    
    -- 接続情報
    connection_config JSONB NOT NULL,
    credentials_encrypted TEXT,
    
    -- 同期設定
    sync_schedule VARCHAR(100), -- Cron式
    last_sync_at TIMESTAMP WITH TIME ZONE,
    next_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(20) DEFAULT 'pending',
    
    -- 統計情報
    total_documents INTEGER DEFAULT 0,
    successful_documents INTEGER DEFAULT 0,
    failed_documents INTEGER DEFAULT 0,
    
    -- 状態管理
    is_active BOOLEAN DEFAULT true,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- タイムスタンプ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_sync_status 
        CHECK (sync_status IN ('pending', 'running', 'completed', 'failed'))
);

-- インデックス
CREATE INDEX idx_sources_type ON sources (source_type);
CREATE INDEX idx_sources_active ON sources (is_active);
CREATE INDEX idx_sources_sync_schedule ON sources (next_sync_at) WHERE is_active = true;
CREATE UNIQUE INDEX idx_sources_name_type ON sources (source_type, name);
```

### 3. 認証・ユーザー管理

#### `users` テーブル
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    
    -- プロフィール
    full_name VARCHAR(255),
    avatar_url TEXT,
    timezone VARCHAR(50) DEFAULT 'Asia/Tokyo',
    
    -- 権限・ロール
    role VARCHAR(50) DEFAULT 'viewer',
    permissions TEXT[], -- PostgreSQL配列
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    
    -- セキュリティ
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- タイムスタンプ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_role 
        CHECK (role IN ('viewer', 'editor', 'admin', 'super_admin')),
    CONSTRAINT valid_email 
        CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- インデックス
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_role ON users (role);
CREATE INDEX idx_users_active ON users (is_active);
CREATE INDEX idx_users_permissions ON users USING GIN (permissions);
```

#### `api_keys` テーブル
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- API Key情報
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL, -- 表示用プレフィックス
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- 権限
    permissions TEXT[] NOT NULL,
    rate_limit_per_minute INTEGER DEFAULT 100,
    
    -- 使用状況
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- 状態
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- タイムスタンプ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT positive_rate_limit CHECK (rate_limit_per_minute > 0)
);

-- インデックス
CREATE INDEX idx_api_keys_user ON api_keys (user_id);
CREATE INDEX idx_api_keys_active ON api_keys (is_active);
CREATE INDEX idx_api_keys_expires ON api_keys (expires_at) WHERE expires_at IS NOT NULL;
```

---

## 🔍 Milvus ベクターコレクション設計

### 1. Dense Vector Collection

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# スキーマ定義
dense_collection_schema = CollectionSchema(
    fields=[
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=255,
            is_primary=True,
            description="チャンクID（PostgreSQLと連携）"
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,  # BGE-M3のDense Vector次元
            description="1024次元Dense Vector"
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="親ドキュメントID"
        ),
        FieldSchema(
            name="source_type",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="ソースタイプ"
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            max_length=2,
            description="言語コード"
        ),
        FieldSchema(
            name="chunk_position",
            dtype=DataType.INT32,
            description="ドキュメント内の位置"
        ),
        FieldSchema(
            name="created_timestamp",
            dtype=DataType.INT64,
            description="作成タイムスタンプ（Unix時間）"
        )
    ],
    description="Dense vectors for semantic search",
    enable_dynamic_field=True  # 将来の拡張に対応
)

# インデックス設定
dense_index_params = {
    "metric_type": "IP",  # Inner Product
    "index_type": "HNSW",
    "params": {
        "M": 16,              # グラフの最大接続数
        "efConstruction": 256  # インデックス構築時の探索幅
    }
}
```

### 2. Sparse Vector Collection

```python
# Sparse Vector用スキーマ
sparse_collection_schema = CollectionSchema(
    fields=[
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=255,
            is_primary=True
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="Sparse Vector（語彙重みマップ）"
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=255
        ),
        FieldSchema(
            name="source_type",
            dtype=DataType.VARCHAR,
            max_length=50
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            max_length=2
        ),
        FieldSchema(
            name="chunk_position",
            dtype=DataType.INT32
        ),
        FieldSchema(
            name="vocabulary_size",
            dtype=DataType.INT32,
            description="有効語彙数"
        ),
        FieldSchema(
            name="created_timestamp",
            dtype=DataType.INT64
        )
    ],
    description="Sparse vectors for keyword search"
)

# Sparse Vector用インデックス
sparse_index_params = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "IP",
    "params": {
        "drop_ratio_build": 0.2  # 低頻度語の除外率
    }
}
```

### 3. Multi-Vector Collection

```python
# Multi-Vector（ColBERT）用スキーマ
multi_collection_schema = CollectionSchema(
    fields=[
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=255,
            is_primary=True
        ),
        FieldSchema(
            name="vectors",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,  # 各トークンベクターの次元
            description="トークンレベルベクターの配列"
        ),
        FieldSchema(
            name="token_count",
            dtype=DataType.INT32,
            description="トークン数"
        ),
        FieldSchema(
            name="token_positions",
            dtype=DataType.JSON,  # トークン位置情報
            description="各トークンの位置情報"
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=255
        ),
        FieldSchema(
            name="source_type",
            dtype=DataType.VARCHAR,
            max_length=50
        ),
        FieldSchema(
            name="chunk_position",
            dtype=DataType.INT32
        ),
        FieldSchema(
            name="created_timestamp",
            dtype=DataType.INT64
        )
    ],
    description="Multi-vectors for fine-grained search"
)
```

---

## 🔗 データ関連付けとマッピング

### PostgreSQL ↔ Milvus 連携

```python
@dataclass
class VectorMappingService:
    """PostgreSQLとMilvusのデータ連携管理"""
    
    async def save_chunk_with_vectors(
        self,
        chunk_data: dict,
        dense_vector: list[float],
        sparse_vector: dict,
        multi_vectors: list[list[float]]
    ) -> str:
        """チャンクデータとベクターを連携保存"""
        
        chunk_id = str(uuid.uuid4())
        
        try:
            # 1. PostgreSQLにチャンクメタデータを保存
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO document_chunks 
                    (id, document_id, content, position, token_count, embedding_status)
                    VALUES ($1, $2, $3, $4, $5, 'processing')
                """, chunk_id, chunk_data["document_id"], 
                    chunk_data["content"], chunk_data["position"], 
                    chunk_data["token_count"])
            
            # 2. Milvusにベクターを保存
            vector_ids = await self._save_vectors_to_milvus(
                chunk_id, dense_vector, sparse_vector, multi_vectors
            )
            
            # 3. PostgreSQLにベクターIDを更新
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE document_chunks 
                    SET dense_vector_id = $1, sparse_vector_id = $2, 
                        multi_vector_id = $3, embedding_status = 'completed',
                        embedded_at = NOW()
                    WHERE id = $4
                """, vector_ids["dense"], vector_ids["sparse"], 
                    vector_ids["multi"], chunk_id)
            
            return chunk_id
            
        except Exception as e:
            # ロールバック処理
            await self._cleanup_failed_chunk(chunk_id)
            raise RuntimeError(f"Chunk saving failed: {e}")
    
    async def _save_vectors_to_milvus(
        self,
        chunk_id: str,
        dense_vector: list[float],
        sparse_vector: dict,
        multi_vectors: list[list[float]]
    ) -> dict[str, str]:
        """Milvusにベクターを保存し、IDを返却"""
        
        vector_ids = {}
        
        # Dense Vector保存
        dense_data = {
            "id": f"{chunk_id}_dense",
            "vector": dense_vector,
            "document_id": chunk_id.split("_")[0],
            "created_timestamp": int(time.time())
        }
        await self.milvus_client.insert("dense_collection", [dense_data])
        vector_ids["dense"] = dense_data["id"]
        
        # Sparse Vector保存
        sparse_data = {
            "id": f"{chunk_id}_sparse",
            "vector": sparse_vector,
            "document_id": chunk_id.split("_")[0],
            "vocabulary_size": len(sparse_vector),
            "created_timestamp": int(time.time())
        }
        await self.milvus_client.insert("sparse_collection", [sparse_data])
        vector_ids["sparse"] = sparse_data["id"]
        
        # Multi-Vector保存
        multi_data = {
            "id": f"{chunk_id}_multi",
            "vectors": multi_vectors,
            "token_count": len(multi_vectors),
            "document_id": chunk_id.split("_")[0],
            "created_timestamp": int(time.time())
        }
        await self.milvus_client.insert("multi_collection", [multi_data])
        vector_ids["multi"] = multi_data["id"]
        
        return vector_ids
```

---

## 📊 データ整合性と制約

### 1. 外部キー制約と削除カスケード

```sql
-- ドキュメント削除時のカスケード処理
CREATE OR REPLACE FUNCTION cleanup_document_vectors()
RETURNS TRIGGER AS $$
DECLARE
    chunk_record RECORD;
BEGIN
    -- 削除されるドキュメントのチャンクIDを取得
    FOR chunk_record IN 
        SELECT dense_vector_id, sparse_vector_id, multi_vector_id 
        FROM document_chunks 
        WHERE document_id = OLD.id
    LOOP
        -- Milvusのベクターを非同期で削除（外部関数経由）
        PERFORM delete_milvus_vectors(
            chunk_record.dense_vector_id,
            chunk_record.sparse_vector_id, 
            chunk_record.multi_vector_id
        );
    END LOOP;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- トリガー設定
CREATE TRIGGER trigger_cleanup_document_vectors
    AFTER DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_document_vectors();
```

### 2. データ状態一貫性チェック

```python
class DataConsistencyChecker:
    """データ整合性チェッカー"""
    
    async def check_consistency(self) -> dict[str, Any]:
        """システム全体の整合性チェック"""
        
        inconsistencies = {
            "orphaned_chunks": [],
            "missing_vectors": [],
            "status_mismatches": [],
            "index_issues": []
        }
        
        # 1. 孤立チャンクチェック
        orphaned = await self._check_orphaned_chunks()
        inconsistencies["orphaned_chunks"] = orphaned
        
        # 2. 欠損ベクターチェック  
        missing_vectors = await self._check_missing_vectors()
        inconsistencies["missing_vectors"] = missing_vectors
        
        # 3. ステータス不整合チェック
        status_issues = await self._check_status_consistency()
        inconsistencies["status_mismatches"] = status_issues
        
        # 4. インデックス整合性チェック
        index_issues = await self._check_index_consistency()
        inconsistencies["index_issues"] = index_issues
        
        return inconsistencies
    
    async def _check_orphaned_chunks(self) -> list[dict]:
        """親ドキュメントのないチャンクを検出"""
        async with self.postgres_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT c.id, c.document_id, c.content
                FROM document_chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE d.id IS NULL
            """)
            return [dict(r) for r in results]
    
    async def _check_missing_vectors(self) -> list[dict]:
        """ベクターが欠損しているチャンクを検出"""
        async with self.postgres_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT id, document_id, embedding_status
                FROM document_chunks 
                WHERE embedding_status = 'completed'
                AND (dense_vector_id IS NULL 
                     OR sparse_vector_id IS NULL 
                     OR multi_vector_id IS NULL)
            """)
            return [dict(r) for r in results]
```

---

## ⚡ パフォーマンス最適化

### 1. パーティショニング戦略

```sql
-- 日付ベースパーティショニング（大量データ対応）
CREATE TABLE documents_partitioned (
    LIKE documents INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- 月別パーティション作成
CREATE TABLE documents_2024_01 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE documents_2024_02 PARTITION OF documents_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 自動パーティション作成関数
CREATE OR REPLACE FUNCTION create_monthly_partition(target_date DATE)
RETURNS void AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    start_date := date_trunc('month', target_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := 'documents_' || to_char(start_date, 'YYYY_MM');
    
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF documents_partitioned
         FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
END;
$$ LANGUAGE plpgsql;
```

### 2. インデックス最適化

```sql
-- 複合インデックス（よく使用される検索条件）
CREATE INDEX idx_chunks_search_optimized 
    ON document_chunks (embedding_status, document_id, position)
    WHERE embedding_status = 'completed';

-- 部分インデックス（アクティブなソースのみ）
CREATE INDEX idx_sources_active_sync 
    ON sources (next_sync_at, source_type)
    WHERE is_active = true;

-- 並行インデックス作成（本番環境対応）
CREATE INDEX CONCURRENTLY idx_documents_fulltext_gin 
    ON documents USING GIN (to_tsvector('japanese', title || ' ' || content));
```

### 3. Milvus パフォーマンス設定

```python
# コレクション作成時の最適化設定
collection_config = {
    "shards_num": 2,          # シャード数
    "consistency_level": "Strong",  # 整合性レベル
}

# インデックス構築パラメータ
index_build_config = {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {
        "M": 16,                    # メモリ効率とのバランス
        "efConstruction": 256,       # 構築時間と精度のバランス
        "max_memory_usage": "2GB"    # メモリ制限
    }
}

# 検索時パラメータ
search_config = {
    "params": {
        "ef": 128,              # 検索精度
        "nprobe": 16            # 並行検索数
    },
    "limit": 100,              # 取得件数上限
    "timeout": 30              # タイムアウト（秒）
}
```

---

## ❗ よくある落とし穴と対策

### 1. ベクター次元不一致

```python
# ❌ 問題: 異なるモデルでの次元数混在
def save_vector_unsafe(vector: list[float]):
    # 次元チェックなしで保存 → 検索時エラー
    collection.insert([{"id": "test", "vector": vector}])

# ✅ 対策: 事前次元チェック
def save_vector_safe(vector: list[float]):
    EXPECTED_DIM = 1024
    if len(vector) != EXPECTED_DIM:
        raise ValueError(
            f"Vector dimension mismatch: expected {EXPECTED_DIM}, "
            f"got {len(vector)}"
        )
    
    # 正規化も実施
    normalized_vector = normalize_vector(vector)
    collection.insert([{"id": "test", "vector": normalized_vector}])
```

### 2. トランザクション境界の問題

```python
# ❌ 問題: 分散データの不整合
async def save_document_unsafe(doc_data, vectors):
    # PostgreSQL保存
    doc_id = await postgres_repo.save(doc_data)
    
    # Milvus保存（失敗時、PostgreSQLデータが残る）
    await milvus_client.insert(vectors)

# ✅ 対策: Sagaパターンによる分散トランザクション
async def save_document_safe(doc_data, vectors):
    compensation_actions = []
    
    try:
        # 1. PostgreSQL保存
        doc_id = await postgres_repo.save(doc_data)
        compensation_actions.append(
            lambda: postgres_repo.delete(doc_id)
        )
        
        # 2. Milvus保存
        vector_ids = await milvus_client.insert(vectors)
        compensation_actions.append(
            lambda: milvus_client.delete(vector_ids)
        )
        
        # 3. マッピング情報更新
        await postgres_repo.update_vector_ids(doc_id, vector_ids)
        
        return doc_id
        
    except Exception as e:
        # 逆順で補償処理実行
        for action in reversed(compensation_actions):
            try:
                await action()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
        raise e
```

### 3. メモリリークとコネクション管理

```python
# ✅ 適切なリソース管理
class DatabaseManager:
    def __init__(self):
        self._postgres_pool = None
        self._milvus_client = None
        self._redis_client = None
    
    async def __aenter__(self):
        # コネクションプール初期化
        self._postgres_pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        self._milvus_client = MilvusClient(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            pool_size=10
        )
        
        self._redis_client = await aioredis.from_url(
            REDIS_URL,
            max_connections=20
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 適切なリソース解放
        if self._postgres_pool:
            await self._postgres_pool.close()
        
        if self._milvus_client:
            await self._milvus_client.close()
        
        if self._redis_client:
            await self._redis_client.close()
```

---

## 🎯 理解確認のための設問

### スキーマ設計理解
1. `documents`テーブルで`processing_status`と`indexing_status`を分けている理由を説明してください
2. `document_chunks`テーブルの`parent_chunk_id`フィールドの用途と階層構造の表現方法を説明してください
3. MilvusでDense、Sparse、Multi-Vectorを別コレクションに分けるメリットを3つ挙げてください

### データ整合性理解
1. PostgreSQL-Milvus間のデータ整合性を保つために実装された仕組みを説明してください
2. 分散トランザクションでSagaパターンを使用する理由と補償処理の重要性を説明してください
3. `cleanup_document_vectors()`トリガー関数が必要な理由を説明してください

### パフォーマンス理解
1. 大量データ処理でパーティショニングが有効な理由と、適切な分割戦略を説明してください
2. Milvusインデックスパラメータ（M、efConstruction）の調整指針を説明してください
3. 部分インデックスを使用することの利点と適用場面を説明してください

### 運用理解
1. データ整合性チェッカーで検出すべき4種類の不整合とその影響を説明してください
2. ベクター次元不一致が発生する原因と事前防止策を説明してください
3. コネクションプール設定で考慮すべきパラメータを5つ挙げてください

---

## 📚 次のステップ

データモデル設計を理解できたら、次の学習段階に進んでください：

- **Step06**: 認証・認可システム - JWT・API Key認証の実装詳細
- **Step07**: エラーハンドリングと監視 - 例外処理・ログ・メトリクス収集
- **Step08**: デプロイメントと運用 - Docker・Kubernetes・CI/CD

適切なデータモデル設計は、システムの拡張性と保守性を決定する重要な要素です。次のステップでは、このデータを安全に保護する認証・認可システムについて学習します。