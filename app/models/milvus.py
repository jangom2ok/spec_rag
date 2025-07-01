"""Milvus用のベクトルコレクション管理"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


class VectorData(BaseModel):
    """ベクトルデータのモデル"""

    id: str = Field(..., description="ユニークID")
    document_id: str = Field(..., description="ドキュメントID")
    chunk_id: str = Field(..., description="チャンクID")

    # Dense vector用
    vector: list[float] | None = Field(None, description="Dense vector")

    # Sparse vector用
    sparse_vector: dict[int, float] | None = Field(None, description="Sparse vector")
    vocabulary_size: int | None = Field(None, description="語彙サイズ")

    # メタデータ
    chunk_type: str | None = Field(None, description="チャンクタイプ")
    source_type: str | None = Field(None, description="ソースタイプ")
    language: str | None = Field("ja", description="言語")
    created_at: int | None = Field(None, description="作成時刻（UnixTime）")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = int(time.time())


class MilvusCollection(ABC):
    """Milvusコレクションの基底クラス"""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.collection_name = self.get_collection_name()
        self.collection: Collection | None = None
        self.connect()

    def connect(self) -> None:
        """Milvusに接続"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            self._initialize_collection()
        except Exception as e:
            print(f"Milvus接続エラー: {e}")
            raise

    def _initialize_collection(self) -> None:
        """コレクションの初期化"""
        if not utility.has_collection(self.collection_name):
            # コレクションが存在しない場合は作成
            schema = self._create_schema()
            self.collection = Collection(name=self.collection_name, schema=schema)

            # インデックスを作成
            index_config = self.get_index_config()
            self.collection.create_index(
                field_name=self.get_vector_field_name(), index_params=index_config
            )
        else:
            # 既存のコレクションを使用
            self.collection = Collection(self.collection_name)

    @abstractmethod
    def get_collection_name(self) -> str:
        """コレクション名を取得"""
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """スキーマ定義を取得"""
        pass

    @abstractmethod
    def get_index_config(self) -> dict[str, Any]:
        """インデックス設定を取得"""
        pass

    @abstractmethod
    def get_vector_field_name(self) -> str:
        """ベクトルフィールド名を取得"""
        pass

    def _create_schema(self) -> CollectionSchema:
        """スキーマを作成"""
        schema_dict = self.get_schema()
        fields = []

        for field_config in schema_dict["fields"]:
            if field_config["type"] == "VARCHAR":
                field = FieldSchema(
                    name=field_config["name"],
                    dtype=DataType.VARCHAR,
                    max_length=field_config["max_length"],
                    is_primary=field_config.get("is_primary", False),
                )
            elif field_config["type"] == "FLOAT_VECTOR":
                field = FieldSchema(
                    name=field_config["name"],
                    dtype=DataType.FLOAT_VECTOR,
                    dim=field_config["dim"],
                )
            elif field_config["type"] == "SPARSE_FLOAT_VECTOR":
                field = FieldSchema(
                    name=field_config["name"], dtype=DataType.SPARSE_FLOAT_VECTOR
                )
            elif field_config["type"] == "INT64":
                field = FieldSchema(name=field_config["name"], dtype=DataType.INT64)
            elif field_config["type"] == "INT32":
                field = FieldSchema(name=field_config["name"], dtype=DataType.INT32)
            else:
                continue

            fields.append(field)

        return CollectionSchema(
            fields=fields, description=f"{schema_dict['name']} collection"
        )

    async def insert(self, data: list[VectorData]) -> dict[str, Any]:
        """データを挿入"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        # データを適切な形式に変換
        insert_data = self._prepare_insert_data(data)

        # 挿入実行
        result = self.collection.insert(insert_data)
        await asyncio.sleep(0)  # async化のため

        # フラッシュしてデータを永続化
        self.collection.flush()

        return {"primary_keys": result.primary_keys}

    async def search(
        self,
        query_vectors: list[list[float] | dict[int, float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ベクトル検索"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        # 検索パラメータ
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # フィルタ式の構築
        expr = self._build_filter_expression(filters) if filters else None

        # 出力フィールドの設定
        if output_fields is None:
            output_fields = ["document_id", "chunk_id", "chunk_type"]

        # 検索実行
        results = self.collection.search(
            data=query_vectors,
            anns_field=self.get_vector_field_name(),
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        await asyncio.sleep(0)  # async化のため

        # 結果を整形
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "ids": [str(hit.id) for hit in result],
                    "distances": [float(hit.distance) for hit in result],
                    "entities": [hit.entity.to_dict() for hit in result],
                }
            )

        return formatted_results

    async def delete_by_document_id(self, document_id: str) -> dict[str, Any]:
        """ドキュメントIDによる削除"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        expr = f"document_id == '{document_id}'"
        result = self.collection.delete(expr)
        await asyncio.sleep(0)  # async化のため

        return {"delete_count": result.delete_count}

    @abstractmethod
    def _prepare_insert_data(self, data: list[VectorData]) -> list[list[Any]]:
        """挿入データの準備"""
        pass

    def _build_filter_expression(self, filters: dict[str, Any]) -> str:
        """フィルタ式を構築"""
        expressions = []
        for key, value in filters.items():
            if isinstance(value, str):
                expressions.append(f"{key} == '{value}'")
            elif isinstance(value, int | float):
                expressions.append(f"{key} == {value}")
            elif isinstance(value, list):
                # IN句の場合
                if all(isinstance(v, str) for v in value):
                    value_str = "', '".join(value)
                    expressions.append(f"{key} in ['{value_str}']")
                else:
                    value_str = ", ".join(str(v) for v in value)
                    expressions.append(f"{key} in [{value_str}]")

        return " and ".join(expressions)


class DenseVectorCollection(MilvusCollection):
    """Dense Vectorコレクション"""

    def get_collection_name(self) -> str:
        return "document_vectors_dense"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "document_vectors_dense",
            "fields": [
                {"name": "id", "type": "VARCHAR", "max_length": 36, "is_primary": True},
                {"name": "document_id", "type": "VARCHAR", "max_length": 36},
                {"name": "chunk_id", "type": "VARCHAR", "max_length": 36},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": 1024},
                {"name": "chunk_type", "type": "VARCHAR", "max_length": 20},
                {"name": "source_type", "type": "VARCHAR", "max_length": 50},
                {"name": "language", "type": "VARCHAR", "max_length": 10},
                {"name": "created_at", "type": "INT64"},
            ],
        }

    def get_index_config(self) -> dict[str, Any]:
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 500},
        }

    def get_vector_field_name(self) -> str:
        return "vector"

    def _prepare_insert_data(self, data: list[VectorData]) -> list[list[Any]]:
        """Dense vector用の挿入データを準備"""
        ids = []
        document_ids = []
        chunk_ids = []
        vectors = []
        chunk_types = []
        source_types = []
        languages = []
        created_ats = []

        for item in data:
            if item.vector is None:
                raise ValueError("Dense vectorが必要です")

            ids.append(item.id)
            document_ids.append(item.document_id)
            chunk_ids.append(item.chunk_id)
            vectors.append(item.vector)
            chunk_types.append(item.chunk_type or "")
            source_types.append(item.source_type or "")
            languages.append(item.language or "ja")
            created_ats.append(item.created_at)

        return [
            ids,
            document_ids,
            chunk_ids,
            vectors,
            chunk_types,
            source_types,
            languages,
            created_ats,
        ]


class SparseVectorCollection(MilvusCollection):
    """Sparse Vectorコレクション"""

    def get_collection_name(self) -> str:
        return "document_vectors_sparse"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "document_vectors_sparse",
            "fields": [
                {"name": "id", "type": "VARCHAR", "max_length": 36, "is_primary": True},
                {"name": "document_id", "type": "VARCHAR", "max_length": 36},
                {"name": "chunk_id", "type": "VARCHAR", "max_length": 36},
                {"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR"},
                {"name": "vocabulary_size", "type": "INT32"},
            ],
        }

    def get_index_config(self) -> dict[str, Any]:
        return {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2},
        }

    def get_vector_field_name(self) -> str:
        return "sparse_vector"

    def _prepare_insert_data(self, data: list[VectorData]) -> list[list[Any]]:
        """Sparse vector用の挿入データを準備"""
        ids = []
        document_ids = []
        chunk_ids = []
        sparse_vectors = []
        vocabulary_sizes = []

        for item in data:
            if item.sparse_vector is None or item.vocabulary_size is None:
                raise ValueError("Sparse vectorとvocabulary_sizeが必要です")

            ids.append(item.id)
            document_ids.append(item.document_id)
            chunk_ids.append(item.chunk_id)
            sparse_vectors.append(item.sparse_vector)
            vocabulary_sizes.append(item.vocabulary_size)

        return [ids, document_ids, chunk_ids, sparse_vectors, vocabulary_sizes]
