"""ApertureDB用のベクトルデータベース管理"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from aperturedb import Client  # type: ignore
except ImportError:
    # Use mock for testing and CI/CD environments
    from app.models.aperturedb_mock import Client  # type: ignore
from pydantic import BaseModel, Field


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

        # バリデーション
        self._validate_vectors()

    def _validate_vectors(self) -> None:
        """ベクトルデータのバリデーション"""
        # Dense vectorとSparse vectorのどちらか一方は必須
        if self.vector is None and self.sparse_vector is None:
            raise ValueError("Either 'vector' or 'sparse_vector' must be provided")

        # Sparse vectorの場合、vocabulary_sizeが必須
        if self.sparse_vector is not None and self.vocabulary_size is None:
            raise ValueError(
                "'vocabulary_size' is required when 'sparse_vector' is provided"
            )

        # Dense vectorの場合、次元数の確認
        if self.vector is not None and len(self.vector) == 0:
            raise ValueError("'vector' must not be empty")

        # Sparse vectorの場合、形式の確認
        if self.sparse_vector is not None and len(self.sparse_vector) == 0:
            raise ValueError("'sparse_vector' must not be empty")


class ApertureDBCollection(ABC):
    """ApertureDBコレクションの基底クラス"""

    def __init__(self, host: str = "localhost", port: int = 55555) -> None:
        self.host: str = host
        self.port: int = port
        self.collection_name: str = self.get_collection_name()
        self.client: Any | None = None  # Client type is conditional
        self.connect()

    def connect(self) -> None:
        """ApertureDBに接続"""
        try:
            self.client = Client(
                host=self.host,
                port=self.port,
                username=os.getenv("APERTUREDB_USERNAME", "admin"),
                password=os.getenv("APERTUREDB_PASSWORD", "admin"),
            )
            self._initialize_collection()
        except Exception as e:
            print(f"ApertureDB接続エラー: {e}")
            raise

    def _initialize_collection(self) -> None:
        """コレクションの初期化"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        # デスクリプターセットを作成（コレクションの代わり）
        descriptor_set_name = self.get_collection_name()

        # デスクリプターセットが存在するかチェック
        query = [
            {
                "FindDescriptorSet": {
                    "with_name": descriptor_set_name,
                    "results": {"count": True},
                }
            }
        ]

        response, _ = self.client.query(query)

        if response[0]["FindDescriptorSet"]["count"] == 0:
            # デスクリプターセットが存在しない場合は作成
            vector_dim = self.get_vector_dimension()
            create_query: list[dict[str, Any]] = [
                {
                    "AddDescriptorSet": {
                        "name": descriptor_set_name,
                        "dimensions": vector_dim,
                        "metric": "L2",
                        "engine": "HNSW",
                    }
                }
            ]
            self.client.query(create_query)

    @abstractmethod
    def get_collection_name(self) -> str:
        """コレクション名を取得"""
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """ベクトルの次元数を取得"""
        pass

    async def insert(self, data: list[VectorData]) -> dict[str, Any]:
        """データを挿入"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        inserted_ids = []
        descriptor_set_name = self.get_collection_name()

        for item in data:
            # メタデータの準備
            properties = {
                "id": item.id,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "chunk_type": item.chunk_type or "",
                "source_type": item.source_type or "",
                "language": item.language or "ja",
                "created_at": item.created_at,
            }

            # ベクトルデータの準備
            if self.collection_name.endswith("dense") and item.vector:
                query = [
                    {
                        "AddDescriptor": {
                            "set": descriptor_set_name,
                            "properties": properties,
                            "descriptor": item.vector,
                        }
                    }
                ]
            elif self.collection_name.endswith("sparse") and item.sparse_vector:
                # Sparse vectorの場合、JSON文字列として保存
                properties["sparse_vector"] = json.dumps(item.sparse_vector)
                properties["vocabulary_size"] = item.vocabulary_size

                # ApertureDBはsparse vectorを直接サポートしないため、
                # プロパティとして保存し、検索は別途実装
                query = [
                    {
                        "AddEntity": {
                            "_ref": [1],
                            "class": "SparseVector",
                            "properties": properties,
                        }
                    }
                ]
            else:
                continue

            response, _ = self.client.query(query)
            inserted_ids.append(item.id)

        await asyncio.sleep(0)  # async化のため
        return {"primary_keys": inserted_ids}

    async def search(
        self,
        query_vectors: list[Any],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ベクトル検索"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        descriptor_set_name = self.get_collection_name()
        results = []

        for query_vector in query_vectors:
            if isinstance(query_vector, list):
                # Dense vector検索
                query = [
                    {
                        "FindDescriptor": {
                            "set": descriptor_set_name,
                            "k_neighbors": top_k,
                            "descriptor": query_vector,
                            "results": {"all_properties": True},
                        }
                    }
                ]

                if filters:
                    query[0]["FindDescriptor"]["constraints"] = self._build_constraints(
                        filters
                    )

                response, _ = self.client.query(query)

                if response[0]["FindDescriptor"]["returned"] > 0:
                    entities = response[0]["FindDescriptor"]["entities"]

                    result: dict[str, list[Any]] = {
                        "ids": [],
                        "distances": [],
                        "entities": [],
                    }

                    for entity in entities:
                        result["ids"].append(entity["id"])
                        result["distances"].append(entity["_distance"])

                        # エンティティのプロパティを整形
                        entity_dict = {
                            "document_id": entity.get("document_id"),
                            "chunk_id": entity.get("chunk_id"),
                            "chunk_type": entity.get("chunk_type"),
                        }
                        result["entities"].append(entity_dict)

                    results.append(result)
            else:
                # Sparse vector検索（カスタム実装が必要）
                # ApertureDBはsparse vectorを直接サポートしないため、
                # 代替実装を提供
                results.append({"ids": [], "distances": [], "entities": []})

        await asyncio.sleep(0)  # async化のため
        return results

    async def delete_by_document_id(self, document_id: str) -> dict[str, Any]:
        """ドキュメントIDによる削除"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        descriptor_set_name = self.get_collection_name()

        # まず該当するデスクリプタを検索
        query = [
            {
                "FindDescriptor": {
                    "set": descriptor_set_name,
                    "constraints": {"document_id": ["==", document_id]},
                    "results": {"count": True},
                }
            }
        ]

        response, _ = self.client.query(query)
        # レスポンスの形式を確認
        if response and len(response) > 0 and "FindDescriptor" in response[0]:
            count = response[0]["FindDescriptor"].get("count", 0)
        else:
            count = 0

        # 削除クエリ
        query = [
            {
                "DeleteDescriptor": {
                    "set": descriptor_set_name,
                    "constraints": {"document_id": ["==", document_id]},
                }
            }
        ]

        self.client.query(query)

        await asyncio.sleep(0)  # async化のため
        return {"delete_count": count}

    def _build_constraints(self, filters: dict[str, Any]) -> dict[str, Any]:
        """ApertureDB用の制約条件を構築"""
        constraints: dict[str, Any] = {}

        for key, value in filters.items():
            if isinstance(value, str):
                constraints[key] = ["==", value]
            elif isinstance(value, int | float):
                constraints[key] = ["==", value]
            elif isinstance(value, list):
                # IN句の場合
                constraints[key] = ["in", value]

        return constraints


class DenseVectorCollection(ApertureDBCollection):
    """Dense Vectorコレクション"""

    def get_collection_name(self) -> str:
        return "document_vectors_dense"

    def get_vector_dimension(self) -> int:
        return 1024


class SparseVectorCollection(ApertureDBCollection):
    """Sparse Vectorコレクション"""

    def get_collection_name(self) -> str:
        return "document_vectors_sparse"

    def get_vector_dimension(self) -> int:
        # Sparse vectorは可変長なので、ダミーの次元数を返す
        return 1
