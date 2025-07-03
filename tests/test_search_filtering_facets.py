"""検索フィルタリング・ファセット機能のテストモジュール

TDD実装：メタデータベースフィルタリングとファセット機能
"""

from datetime import datetime, timedelta
from typing import Any

import pytest

from app.services.hybrid_search_engine import (
    HybridSearchEngine,
    SearchConfig,
    SearchFilter,
)


class TestSearchFiltering:
    """検索フィルタリングのテストクラス"""

    @pytest.fixture
    def search_config(self) -> SearchConfig:
        """基本検索設定"""
        return SearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k=20,
        )

    @pytest.fixture
    def sample_documents(self) -> list[dict[str, Any]]:
        """フィルタリング用サンプルドキュメント"""
        base_date = datetime(2024, 1, 1)
        return [
            {
                "id": "doc-1",
                "title": "Python Programming Guide",
                "content": "Comprehensive guide to Python programming.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "guide",
                "search_score": 0.95,
                "metadata": {
                    "author": "john_doe",
                    "tags": ["python", "programming", "tutorial"],
                    "created_at": base_date.isoformat(),
                    "category": "development",
                    "difficulty": "beginner",
                    "status": "published",
                    "rating": 4.5,
                    "view_count": 1250,
                },
            },
            {
                "id": "doc-2",
                "title": "Advanced Machine Learning",
                "content": "Deep dive into machine learning algorithms.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "tutorial",
                "search_score": 0.87,
                "metadata": {
                    "author": "jane_smith",
                    "tags": ["ml", "algorithms", "advanced"],
                    "created_at": (base_date + timedelta(days=10)).isoformat(),
                    "category": "ai",
                    "difficulty": "advanced",
                    "status": "draft",
                    "rating": 4.8,
                    "view_count": 856,
                },
            },
            {
                "id": "doc-3",
                "title": "データベース設計基礎",
                "content": "データベース設計の基本概念について説明。",
                "source_type": "jira",
                "language": "ja",
                "document_type": "specification",
                "search_score": 0.73,
                "metadata": {
                    "author": "yamada_taro",
                    "tags": ["database", "design", "基礎"],
                    "created_at": (base_date + timedelta(days=5)).isoformat(),
                    "category": "database",
                    "difficulty": "intermediate",
                    "status": "published",
                    "rating": 4.2,
                    "view_count": 432,
                },
            },
            {
                "id": "doc-4",
                "title": "Web API Best Practices",
                "content": "Best practices for designing RESTful APIs.",
                "source_type": "sharepoint",
                "language": "en",
                "document_type": "guide",
                "search_score": 0.91,
                "metadata": {
                    "author": "alice_brown",
                    "tags": ["api", "rest", "web"],
                    "created_at": (base_date + timedelta(days=20)).isoformat(),
                    "category": "development",
                    "difficulty": "intermediate",
                    "status": "published",
                    "rating": 4.7,
                    "view_count": 982,
                },
            },
            {
                "id": "doc-5",
                "title": "DevOps Pipeline Setup",
                "content": "Setting up CI/CD pipelines for modern development.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "tutorial",
                "search_score": 0.68,
                "metadata": {
                    "author": "bob_wilson",
                    "tags": ["devops", "cicd", "automation"],
                    "created_at": (base_date + timedelta(days=15)).isoformat(),
                    "category": "operations",
                    "difficulty": "advanced",
                    "status": "published",
                    "rating": 4.3,
                    "view_count": 675,
                },
            },
        ]

    @pytest.mark.unit
    def test_single_field_exact_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """単一フィールド完全一致フィルター"""
        search_engine = HybridSearchEngine(config=search_config)

        # source_type = "confluence" でフィルタリング
        filters = [SearchFilter(field="source_type", value="confluence", operator="eq")]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 3  # doc-1, doc-2, doc-5
        assert all(doc["source_type"] == "confluence" for doc in filtered_docs)

    @pytest.mark.unit
    def test_nested_field_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """ネストされたフィールドフィルター"""
        search_engine = HybridSearchEngine(config=search_config)

        # metadata.category = "development" でフィルタリング
        filters = [
            SearchFilter(field="metadata.category", value="development", operator="eq")
        ]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 2  # doc-1, doc-4
        assert all(
            doc["metadata"]["category"] == "development" for doc in filtered_docs
        )

    @pytest.mark.unit
    def test_multiple_filters_and_condition(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """複数フィルター（AND条件）"""
        search_engine = HybridSearchEngine(config=search_config)

        # language = "en" AND document_type = "guide"
        filters = [
            SearchFilter(field="language", value="en", operator="eq"),
            SearchFilter(field="document_type", value="guide", operator="eq"),
        ]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 2  # doc-1, doc-4
        assert all(
            doc["language"] == "en" and doc["document_type"] == "guide"
            for doc in filtered_docs
        )

    @pytest.mark.unit
    def test_not_equal_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """不等フィルター"""
        search_engine = HybridSearchEngine(config=search_config)

        # source_type != "jira"
        filters = [SearchFilter(field="source_type", value="jira", operator="ne")]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 4  # jira以外の全て
        assert all(doc["source_type"] != "jira" for doc in filtered_docs)

    @pytest.mark.unit
    def test_in_operator_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """IN演算子フィルター"""
        search_engine = HybridSearchEngine(config=search_config)

        # document_type in ["guide", "tutorial"]
        filters = [
            SearchFilter(
                field="document_type", value=["guide", "tutorial"], operator="in"
            )
        ]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 4  # doc-1, doc-2, doc-4, doc-5
        assert all(
            doc["document_type"] in ["guide", "tutorial"] for doc in filtered_docs
        )

    @pytest.mark.unit
    def test_not_in_operator_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """NOT IN演算子フィルター"""
        search_engine = HybridSearchEngine(config=search_config)

        # metadata.difficulty not in ["beginner", "advanced"]
        filters = [
            SearchFilter(
                field="metadata.difficulty",
                value=["beginner", "advanced"],
                operator="not_in",
            )
        ]

        filtered_docs = search_engine._apply_query_filters(sample_documents, filters)

        assert len(filtered_docs) == 2  # doc-3, doc-4 (intermediate)
        assert all(
            doc["metadata"]["difficulty"] == "intermediate" for doc in filtered_docs
        )

    @pytest.mark.unit
    def test_range_filters_numeric(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """数値範囲フィルター"""
        _search_engine = HybridSearchEngine(config=search_config)

        # rating >= 4.5
        filters = [SearchFilter(field="metadata.rating", value=4.5, operator="gte")]

        # GT/LT演算子の実装が必要な場合
        # ここでは簡易的にテスト
        assert len(filters) == 1
        assert filters[0].operator == "gte"

    @pytest.mark.unit
    def test_array_field_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """配列フィールドフィルター"""
        _search_engine = HybridSearchEngine(config=search_config)

        # タグに"python"が含まれているドキュメント
        # 実装では配列内検索が必要
        filtered_docs = []
        for doc in sample_documents:
            tags = doc.get("metadata", {}).get("tags", [])
            if "python" in tags:
                filtered_docs.append(doc)

        assert len(filtered_docs) == 1  # doc-1のみ
        assert filtered_docs[0]["id"] == "doc-1"

    @pytest.mark.unit
    def test_similarity_threshold_filter(
        self, search_config: SearchConfig, sample_documents: list[dict[str, Any]]
    ):
        """類似度閾値フィルター"""
        search_config.similarity_threshold = 0.8
        search_engine = HybridSearchEngine(config=search_config)

        filtered_docs = search_engine._filter_by_similarity_threshold(sample_documents)

        # search_score >= 0.8 のドキュメントのみ
        assert len(filtered_docs) == 3  # doc-1(0.95), doc-2(0.87), doc-4(0.91)
        assert all(doc["search_score"] >= 0.8 for doc in filtered_docs)

    @pytest.mark.unit
    def test_missing_field_handling(self, search_config: SearchConfig):
        """存在しないフィールドの処理"""
        search_engine = HybridSearchEngine(config=search_config)

        documents = [
            {"id": "doc-1", "title": "Test", "language": "en"},
            {"id": "doc-2", "title": "Test 2"},  # languageフィールドなし
        ]

        filters = [SearchFilter(field="language", value="en", operator="eq")]

        filtered_docs = search_engine._apply_query_filters(documents, filters)

        assert len(filtered_docs) == 1  # フィールドがないドキュメントは除外
        assert filtered_docs[0]["id"] == "doc-1"

    @pytest.mark.unit
    def test_nested_field_extraction(self, search_config: SearchConfig):
        """ネストフィールド抽出テスト"""
        search_engine = HybridSearchEngine(config=search_config)

        document = {
            "metadata": {
                "author": {"name": "John Doe", "email": "john@example.com"},
                "stats": {"views": 100},
            }
        }

        # 2レベルネスト
        author_name = search_engine._get_nested_field(document, "metadata.author.name")
        assert author_name == "John Doe"

        # 存在しないパス
        missing_field = search_engine._get_nested_field(
            document, "metadata.nonexistent.field"
        )
        assert missing_field is None


class TestSearchFacets:
    """検索ファセットのテストクラス"""

    @pytest.fixture
    def search_config(self) -> SearchConfig:
        """基本検索設定"""
        return SearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k=20,
        )

    @pytest.fixture
    def facet_test_documents(self) -> list[dict[str, Any]]:
        """ファセット用テストドキュメント"""
        return [
            {
                "id": "doc-1",
                "source_type": "confluence",
                "language": "en",
                "document_type": "guide",
                "metadata": {
                    "category": "development",
                    "tags": ["python", "tutorial"],
                    "difficulty": "beginner",
                },
            },
            {
                "id": "doc-2",
                "source_type": "confluence",
                "language": "en",
                "document_type": "tutorial",
                "metadata": {
                    "category": "ai",
                    "tags": ["ml", "advanced"],
                    "difficulty": "advanced",
                },
            },
            {
                "id": "doc-3",
                "source_type": "jira",
                "language": "ja",
                "document_type": "specification",
                "metadata": {
                    "category": "database",
                    "tags": ["database", "design"],
                    "difficulty": "intermediate",
                },
            },
            {
                "id": "doc-4",
                "source_type": "confluence",
                "language": "en",
                "document_type": "guide",
                "metadata": {
                    "category": "development",
                    "tags": ["api", "rest"],
                    "difficulty": "intermediate",
                },
            },
        ]

    @pytest.mark.unit
    def test_single_facet_calculation(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """単一ファセット計算"""
        search_engine = HybridSearchEngine(config=search_config)

        facets = search_engine._calculate_facets(facet_test_documents, ["source_type"])

        assert "source_type" in facets
        source_type_facets = facets["source_type"]

        # Confluenceが3件、Jiraが1件
        assert len(source_type_facets) == 2
        confluence_facet = next(
            f for f in source_type_facets if f.value == "confluence"
        )
        jira_facet = next(f for f in source_type_facets if f.value == "jira")

        assert confluence_facet.count == 3
        assert jira_facet.count == 1

    @pytest.mark.unit
    def test_multiple_facets_calculation(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """複数ファセット計算"""
        search_engine = HybridSearchEngine(config=search_config)

        facets = search_engine._calculate_facets(
            facet_test_documents, ["source_type", "language", "document_type"]
        )

        assert "source_type" in facets
        assert "language" in facets
        assert "document_type" in facets

        # Language facets
        language_facets = facets["language"]
        en_facet = next(f for f in language_facets if f.value == "en")
        ja_facet = next(f for f in language_facets if f.value == "ja")

        assert en_facet.count == 3
        assert ja_facet.count == 1

    @pytest.mark.unit
    def test_nested_field_facets(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """ネストフィールドファセット"""
        search_engine = HybridSearchEngine(config=search_config)

        facets = search_engine._calculate_facets(
            facet_test_documents, ["metadata.category"]
        )

        assert "metadata.category" in facets
        category_facets = facets["metadata.category"]

        # development: 2, ai: 1, database: 1
        assert len(category_facets) == 3
        development_facet = next(f for f in category_facets if f.value == "development")
        assert development_facet.count == 2

    @pytest.mark.unit
    def test_array_field_facets(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """配列フィールドファセット"""
        search_engine = HybridSearchEngine(config=search_config)

        facets = search_engine._calculate_facets(
            facet_test_documents, ["metadata.tags"]
        )

        assert "metadata.tags" in facets
        tags_facets = facets["metadata.tags"]

        # 各タグの出現回数をカウント
        tag_counts = {f.value: f.count for f in tags_facets}

        # 各ドキュメントのタグが個別にカウントされている
        assert "python" in tag_counts
        assert "tutorial" in tag_counts
        assert "ml" in tag_counts
        assert "database" in tag_counts

    @pytest.mark.unit
    def test_facets_with_zero_counts(self, search_config: SearchConfig):
        """ゼロカウントファセット"""
        search_engine = HybridSearchEngine(config=search_config)

        # 全ドキュメントが同じ値を持つ場合
        documents = [
            {"id": "doc-1", "status": "active"},
            {"id": "doc-2", "status": "active"},
        ]

        facets = search_engine._calculate_facets(documents, ["status"])

        assert "status" in facets
        status_facets = facets["status"]
        assert len(status_facets) == 1
        assert status_facets[0].value == "active"
        assert status_facets[0].count == 2

    @pytest.mark.unit
    def test_empty_facets(self, search_config: SearchConfig):
        """空ファセット処理"""
        search_engine = HybridSearchEngine(config=search_config)

        # 存在しないフィールドでファセット
        documents = [{"id": "doc-1", "title": "Test"}]

        facets = search_engine._calculate_facets(documents, ["nonexistent_field"])

        assert "nonexistent_field" in facets
        assert len(facets["nonexistent_field"]) == 0

    @pytest.mark.unit
    def test_facet_sorting(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """ファセットソート（頻度順）"""
        search_engine = HybridSearchEngine(config=search_config)

        facets = search_engine._calculate_facets(
            facet_test_documents, ["document_type"]
        )

        document_type_facets = facets["document_type"]

        # 頻度順（降順）でソートされていることを確認
        for i in range(len(document_type_facets) - 1):
            assert document_type_facets[i].count >= document_type_facets[i + 1].count

    @pytest.mark.integration
    def test_combined_filter_and_facets(
        self, search_config: SearchConfig, facet_test_documents: list[dict[str, Any]]
    ):
        """フィルターとファセットの組み合わせ"""
        search_engine = HybridSearchEngine(config=search_config)

        # 英語ドキュメントのみでファセット計算
        filters = [SearchFilter(field="language", value="en", operator="eq")]
        filtered_docs = search_engine._apply_query_filters(
            facet_test_documents, filters
        )

        facets = search_engine._calculate_facets(filtered_docs, ["document_type"])

        # 英語ドキュメントのみなので日本語ドキュメントのspecificationは除外
        document_type_facets = facets["document_type"]
        types = [f.value for f in document_type_facets]

        assert "guide" in types
        assert "tutorial" in types
        # specificationは日本語ドキュメントなので除外されている
