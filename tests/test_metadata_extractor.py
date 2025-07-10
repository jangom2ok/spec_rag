"""メタデータ抽出器のテストモジュール

TDD実装：テストケース→実装→リファクタの順序で実装
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from app.services.metadata_extractor import (
    DocumentType,
    ExtractionConfig,
    ExtractionResult,
    FieldType,
    MetadataExtractor,
)

# Use extended fixtures
# Use extended fixtures from conftest.py


class TestMetadataExtractor:
    """メタデータ抽出器のテストクラス"""

    @pytest.fixture
    def basic_config(self) -> ExtractionConfig:
        """基本的な抽出設定"""
        return ExtractionConfig(
            extract_structure=True,
            extract_entities=True,
            extract_keywords=True,
            extract_statistics=True,
            language_detection=True,
            confidence_threshold=0.7,
        )

    @pytest.fixture
    def minimal_config(self) -> ExtractionConfig:
        """最小限の抽出設定"""
        return ExtractionConfig(
            extract_structure=False,
            extract_entities=False,
            extract_keywords=False,
            extract_statistics=True,
            language_detection=True,
        )

    @pytest.fixture
    def sample_markdown_document(self) -> dict[str, Any]:
        """Markdownサンプルドキュメント"""
        return {
            "id": "md-doc-1",
            "title": "Technical Documentation",
            "content": """# API Documentation

## Overview
This document describes the REST API for our system.

### Authentication
All endpoints require authentication via API key.

#### API Key Format
- Format: `api_key_xxxxxxxxxxxxxxxx`
- Length: 32 characters
- Scope: read, write, admin

### Endpoints

#### GET /users
Returns a list of users.

**Parameters:**
- `limit` (integer): Maximum number of results
- `offset` (integer): Pagination offset

**Response:**
```json
{
  "users": [
    {"id": 1, "name": "John Doe", "email": "john@example.com"}
  ],
  "total": 100,
  "page": 1
}
```

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error""",
            "source_type": "markdown",
            "file_type": "md",
            "metadata": {"author": "tech_writer", "version": "1.0"},
        }

    @pytest.fixture
    def sample_confluence_document(self) -> dict[str, Any]:
        """Confluenceサンプルドキュメント"""
        return {
            "id": "conf-doc-1",
            "title": "System Requirements",
            "content": """<h1>System Requirements</h1>

<h2>Hardware Requirements</h2>
<ul>
<li>CPU: Intel i5 or equivalent</li>
<li>RAM: 8GB minimum, 16GB recommended</li>
<li>Storage: 500GB SSD</li>
<li>Network: 1Gbps connection</li>
</ul>

<h2>Software Requirements</h2>
<p>The following software must be installed:</p>
<ul>
<li>Operating System: Ubuntu 20.04 LTS or CentOS 8</li>
<li>Docker: Version 20.10 or later</li>
<li>Python: Version 3.9 or later</li>
<li>PostgreSQL: Version 13 or later</li>
</ul>

<h2>Network Configuration</h2>
<p>Configure the following ports:</p>
<table>
<tr><th>Service</th><th>Port</th><th>Protocol</th></tr>
<tr><td>Web Server</td><td>80, 443</td><td>HTTP/HTTPS</td></tr>
<tr><td>Database</td><td>5432</td><td>TCP</td></tr>
<tr><td>API</td><td>8000</td><td>HTTP</td></tr>
</table>""",
            "source_type": "confluence",
            "file_type": "html",
            "metadata": {
                "space": "TECH",
                "page_id": "123456",
                "author": "admin",
                "labels": ["requirements", "system", "infrastructure"],
            },
        }

    @pytest.fixture
    def sample_japanese_document(self) -> dict[str, Any]:
        """日本語サンプルドキュメント"""
        return {
            "id": "jp-doc-1",
            "title": "システム仕様書",
            "content": """# システム仕様書

## 概要
本システムは、企業向けのドキュメント管理システムです。

## 機能要件

### 1. ユーザー管理機能
- ユーザー登録・削除
- 権限管理（管理者、編集者、閲覧者）
- ログイン・ログアウト

### 2. ドキュメント管理機能
- ドキュメントのアップロード・ダウンロード
- バージョン管理
- 検索機能（全文検索、タグ検索）

### 3. セキュリティ機能
- SSL/TLS暗号化
- アクセス制御
- 監査ログ

## 非機能要件

### パフォーマンス
- 応答時間：1秒以内
- 同時接続数：1000ユーザー
- 稼働率：99.9%以上

### セキュリティ
- データ暗号化（AES-256）
- 定期的なセキュリティ監査
- 脆弱性スキャン

## 技術仕様

### アーキテクチャ
- マイクロサービス アーキテクチャ
- コンテナ化（Docker）
- オーケストレーション（Kubernetes）

### 使用技術
- フロントエンド：React.js、TypeScript
- バックエンド：Python、FastAPI
- データベース：PostgreSQL、Redis
- インフラ：AWS、CloudFormation""",
            "source_type": "markdown",
            "file_type": "md",
            "metadata": {
                "author": "system_analyst",
                "department": "IT",
                "version": "2.1",
            },
        }

    @pytest.mark.unit
    async def test_extractor_initialization(self, basic_config: ExtractionConfig):
        """抽出器の初期化テスト"""
        extractor = MetadataExtractor(config=basic_config)

        assert extractor.config == basic_config
        assert extractor.config.extract_structure is True
        assert extractor.config.extract_entities is True
        assert extractor.config.confidence_threshold == 0.7

    @pytest.mark.unit
    async def test_basic_metadata_extraction(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any]
    ):
        """基本メタデータ抽出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        assert isinstance(result, ExtractionResult)
        assert result.success is True
        assert result.document_id == sample_markdown_document["id"]

        # 基本統計情報の確認
        assert "statistics" in result.metadata
        stats = result.metadata["statistics"]
        assert "word_count" in stats
        assert "char_count" in stats
        assert "line_count" in stats
        assert stats["word_count"] > 0
        assert stats["char_count"] > 0

    @pytest.mark.unit
    async def test_document_type_detection(self, basic_config: ExtractionConfig):
        """ドキュメントタイプ検出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        # Markdownドキュメント
        md_doc = {
            "id": "md-1",
            "content": "# Title\n\nSome content with **bold** text.",
            "file_type": "md",
        }
        result = await extractor.extract_metadata(md_doc)
        assert result.metadata["document_type"] == DocumentType.MARKDOWN

        # HTMLドキュメント
        html_doc = {
            "id": "html-1",
            "content": "<h1>Title</h1><p>Some content with <strong>bold</strong> text.</p>",
            "file_type": "html",
        }
        result = await extractor.extract_metadata(html_doc)
        assert result.metadata["document_type"] == DocumentType.HTML

        # プレーンテキスト
        text_doc = {
            "id": "txt-1",
            "content": "Simple plain text document.",
            "file_type": "txt",
        }
        result = await extractor.extract_metadata(text_doc)
        assert result.metadata["document_type"] == DocumentType.PLAIN_TEXT

    @pytest.mark.unit
    async def test_language_detection(
        self, basic_config: ExtractionConfig, sample_japanese_document: dict[str, Any]
    ):
        """言語検出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_japanese_document)

        assert "language" in result.metadata
        assert result.metadata["language"]["detected"] == "ja"
        assert result.metadata["language"]["confidence"] > 0.8

    @pytest.mark.unit
    async def test_structure_extraction_markdown(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any]
    ):
        """Markdown構造抽出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        assert "structure" in result.metadata
        structure = result.metadata["structure"]

        # 見出し構造の確認
        assert "headings" in structure
        headings = structure["headings"]
        assert len(headings) > 0

        # 最初の見出しがh1であることを確認
        assert headings[0]["level"] == 1
        assert "API Documentation" in headings[0]["text"]

        # 階層構造の確認
        h2_headings = [h for h in headings if h["level"] == 2]
        assert len(h2_headings) >= 2  # "Overview", "Error Handling"など

    @pytest.mark.unit
    async def test_structure_extraction_html(
        self, basic_config: ExtractionConfig, sample_confluence_document: dict[str, Any]
    ):
        """HTML構造抽出テスト"""
        # Mock the HTML parser if needed
        with patch("app.services.metadata_extractor.BeautifulSoup") as mock_bs:
            # Setup mock HTML parsing
            mock_soup = Mock()
            mock_soup.find_all.side_effect = lambda tag: [
                Mock(name=tag, get_text=lambda: f"Sample {tag} content")
                for _ in range(2 if tag in ["ul", "ol", "table"] else 0)
            ]
            mock_bs.return_value = mock_soup
            
            extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_confluence_document)

        assert "structure" in result.metadata
        structure = result.metadata["structure"]

        # HTML要素の確認
        assert "elements" in structure
        elements = structure["elements"]

        # リスト要素の検出
        lists = [e for e in elements if e["type"] == "list"]
        assert len(lists) > 0

        # テーブル要素の検出
        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) > 0

    @pytest.mark.unit
    async def test_entity_extraction(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any], mock_spacy_model
    ):
        """エンティティ抽出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        assert "entities" in result.metadata
        entities = result.metadata["entities"]

        # 技術的なエンティティの検出
        assert "technical_terms" in entities
        tech_terms = entities["technical_terms"]

        # APIキーフォーマットの検出
        assert any("api_key" in term.lower() for term in tech_terms)

        # HTTPステータスコードの検出
        assert "http_codes" in entities
        http_codes = entities["http_codes"]
        assert "200" in http_codes
        assert "404" in http_codes

    @pytest.mark.unit
    async def test_keyword_extraction(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any], mock_spacy_model
    ):
        """キーワード抽出テスト"""
        # Mock keyword extraction
        with patch("app.services.metadata_extractor.extract_keywords") as mock_extract:
            mock_extract.return_value = [
                {"text": "API", "score": 0.9},
                {"text": "authentication", "score": 0.85},
                {"text": "REST", "score": 0.8},
            ]
            
            extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        assert "keywords" in result.metadata
        keywords = result.metadata["keywords"]

        # 重要キーワードの確認
        assert len(keywords) > 0
        keyword_texts = [kw["text"] for kw in keywords]

        # API関連のキーワードが含まれていることを確認
        assert any("api" in kw.lower() for kw in keyword_texts)
        assert any("authentication" in kw.lower() for kw in keyword_texts)

    @pytest.mark.unit
    async def test_statistics_extraction(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any]
    ):
        """統計情報抽出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        assert "statistics" in result.metadata
        stats = result.metadata["statistics"]

        # 基本統計
        assert "word_count" in stats
        assert "char_count" in stats
        assert "line_count" in stats
        assert "paragraph_count" in stats

        # 文書複雑度
        assert "complexity" in stats
        complexity = stats["complexity"]
        assert "readability_score" in complexity
        assert "sentence_length_avg" in complexity

        # 構造統計
        assert "structure_stats" in stats
        structure_stats = stats["structure_stats"]
        assert "heading_count" in structure_stats
        assert "list_count" in structure_stats

    @pytest.mark.unit
    async def test_confidence_filtering(self):
        """信頼度フィルタリングテスト"""
        high_confidence_config = ExtractionConfig(
            extract_entities=True, confidence_threshold=0.9
        )

        low_confidence_config = ExtractionConfig(
            extract_entities=True, confidence_threshold=0.3
        )

        document = {
            "id": "test-1",
            "content": "This is a simple test document with some API endpoints.",
        }

        high_extractor = MetadataExtractor(config=high_confidence_config)
        low_extractor = MetadataExtractor(config=low_confidence_config)

        high_result = await high_extractor.extract_metadata(document)
        low_result = await low_extractor.extract_metadata(document)

        # 低い信頼度設定の方がより多くの結果を返すはず
        if "entities" in high_result.metadata and "entities" in low_result.metadata:
            high_entities = len(
                high_result.metadata["entities"].get("technical_terms", [])
            )
            low_entities = len(
                low_result.metadata["entities"].get("technical_terms", [])
            )
            assert low_entities >= high_entities

    @pytest.mark.unit
    async def test_empty_document_handling(self, basic_config: ExtractionConfig):
        """空ドキュメントの処理テスト"""
        extractor = MetadataExtractor(config=basic_config)

        empty_document = {
            "id": "empty-1",
            "content": "",
        }

        result = await extractor.extract_metadata(empty_document)

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @pytest.mark.unit
    async def test_large_document_handling(self, basic_config: ExtractionConfig, mock_spacy_model):
        """大きなドキュメントの処理テスト"""
        # Disable NLP processing for large documents to avoid memory issues
        config = ExtractionConfig(
            extract_structure=False,
            extract_entities=False,
            extract_keywords=False,
            extract_statistics=True,
            language_detection=False,
        )
        extractor = MetadataExtractor(config=config)

        # 大きなドキュメントを生成
        large_content = "This is a test sentence. " * 10000  # 約10万語
        large_document = {
            "id": "large-1",
            "content": large_content,
        }

        result = await extractor.extract_metadata(large_document)

        assert result.success is True
        assert result.metadata["statistics"]["word_count"] > 50000

    @pytest.mark.unit
    async def test_metadata_field_types(
        self, basic_config: ExtractionConfig, sample_markdown_document: dict[str, Any]
    ):
        """メタデータフィールドタイプテスト"""
        extractor = MetadataExtractor(config=basic_config)

        result = await extractor.extract_metadata(sample_markdown_document)

        # フィールドタイプの確認
        field_types = result.field_types

        assert field_types["statistics"] == FieldType.OBJECT
        assert field_types["language"] == FieldType.OBJECT
        assert field_types["document_type"] == FieldType.STRING

        if "keywords" in field_types:
            assert field_types["keywords"] == FieldType.ARRAY

    @pytest.mark.unit
    async def test_incremental_extraction(self, basic_config: ExtractionConfig):
        """増分抽出テスト"""
        extractor = MetadataExtractor(config=basic_config)

        # 既存のメタデータがある場合
        document_with_metadata = {
            "id": "incremental-1",
            "content": "New content for incremental extraction.",
            "existing_metadata": {
                "custom_field": "custom_value",
                "previous_analysis": {"version": "1.0"},
            },
        }

        result = await extractor.extract_metadata(document_with_metadata)

        # 既存のメタデータが保持されていることを確認
        if "existing_metadata" in result.metadata:
            assert (
                result.metadata["existing_metadata"]["custom_field"] == "custom_value"
            )


class TestExtractionConfig:
    """抽出設定のテストクラス"""

    @pytest.mark.unit
    def test_config_creation(self):
        """設定作成テスト"""
        config = ExtractionConfig(
            extract_structure=True, extract_entities=False, confidence_threshold=0.8
        )

        assert config.extract_structure is True
        assert config.extract_entities is False
        assert config.confidence_threshold == 0.8

    @pytest.mark.unit
    def test_config_validation(self):
        """設定バリデーションテスト"""
        # 有効な設定
        valid_config = ExtractionConfig(confidence_threshold=0.5)
        assert valid_config.confidence_threshold == 0.5

        # 無効な設定（信頼度が範囲外）
        with pytest.raises(ValueError):
            ExtractionConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            ExtractionConfig(confidence_threshold=-0.1)


class TestExtractionResult:
    """抽出結果のテストクラス"""

    @pytest.mark.unit
    def test_result_creation(self):
        """結果作成テスト"""
        metadata = {
            "statistics": {"word_count": 100},
            "language": {"detected": "en"},
        }

        field_types = {
            "statistics": FieldType.OBJECT,
            "language": FieldType.OBJECT,
        }

        result = ExtractionResult(
            success=True,
            document_id="test-1",
            metadata=metadata,
            field_types=field_types,
            processing_time=1.5,
        )

        assert result.success is True
        assert result.document_id == "test-1"
        assert result.metadata["statistics"]["word_count"] == 100
        assert result.field_types["statistics"] == FieldType.OBJECT

    @pytest.mark.unit
    def test_result_summary(self):
        """結果サマリーテスト"""
        result = ExtractionResult(
            success=True,
            document_id="test-1",
            metadata={"field1": "value1", "field2": {"nested": "value"}},
            field_types={"field1": FieldType.STRING, "field2": FieldType.OBJECT},
            processing_time=2.0,
        )

        summary = result.get_summary()

        assert "total_fields" in summary
        assert "processing_time" in summary
        assert "success" in summary
        assert summary["total_fields"] == 2
        assert summary["processing_time"] == 2.0
        assert summary["success"] is True
