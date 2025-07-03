"""メタデータ抽出サービス

TDD実装：テストケースに基づいたメタデータ抽出機能
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """ドキュメントタイプ"""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    CONFLUENCE = "confluence"
    JIRA = "jira"


class FieldType(str, Enum):
    """メタデータフィールドタイプ"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATETIME = "datetime"


@dataclass
class MetadataField:
    """メタデータフィールド定義"""

    name: str
    field_type: FieldType
    description: str
    required: bool = False
    confidence: float = 1.0


@dataclass
class ExtractionConfig:
    """メタデータ抽出設定"""

    extract_structure: bool = True
    extract_entities: bool = True
    extract_keywords: bool = True
    extract_statistics: bool = True
    language_detection: bool = True
    confidence_threshold: float = 0.5
    max_keywords: int = 20
    max_entities: int = 50

    def __post_init__(self):
        """設定値のバリデーション"""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.max_keywords <= 0:
            raise ValueError("max_keywords must be greater than 0")
        if self.max_entities <= 0:
            raise ValueError("max_entities must be greater than 0")


@dataclass
class ExtractionResult:
    """メタデータ抽出結果"""

    success: bool
    document_id: str
    metadata: dict[str, Any]
    field_types: dict[str, FieldType]
    processing_time: float
    error_message: str | None = None
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_summary(self) -> dict[str, Any]:
        """抽出結果のサマリーを取得"""
        return {
            "success": self.success,
            "document_id": self.document_id,
            "total_fields": len(self.metadata),
            "processing_time": self.processing_time,
            "extracted_at": self.extracted_at,
        }


class LanguageDetector:
    """言語検出ユーティリティ"""

    @staticmethod
    def detect_language(text: str) -> tuple[str, float]:
        """簡易的な言語検出"""
        # 日本語文字の検出
        japanese_chars = len(
            re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text)
        )
        total_chars = len(re.findall(r"[^\s]", text))

        if total_chars == 0:
            return "unknown", 0.0

        japanese_ratio = japanese_chars / total_chars

        if japanese_ratio > 0.1:
            return "ja", min(japanese_ratio * 2, 1.0)
        else:
            return "en", max(0.8, 1.0 - japanese_ratio)


class StructureExtractor:
    """文書構造抽出器"""

    @staticmethod
    def extract_markdown_structure(content: str) -> dict[str, Any]:
        """Markdown構造を抽出"""
        structure: dict[str, Any] = {
            "headings": [],
            "lists": [],
            "code_blocks": [],
            "links": [],
        }

        lines = content.split("\n")

        # 見出し抽出
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                if text:
                    structure["headings"].append(
                        {
                            "level": level,
                            "text": text,
                            "line": i + 1,
                        }
                    )

        # リスト抽出
        in_list = False
        current_list: list[str] = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(("-", "*", "+")) or re.match(r"^\d+\.", line):
                if not in_list:
                    in_list = True
                    current_list = []
                current_list.append(
                    {
                        "text": re.sub(r"^[-*+]|\d+\.", "", line).strip(),
                        "line": i + 1,
                    }
                )
            else:
                if in_list and current_list:
                    structure["lists"].append(current_list)
                    current_list = []
                in_list = False

        # 最後のリストを追加
        if current_list:
            structure["lists"].append(current_list)

        # コードブロック抽出
        in_code_block = False
        current_code: list[str] = []

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if in_code_block:
                    structure["code_blocks"].append(
                        {
                            "content": "\n".join(current_code),
                            "language": current_code[0] if current_code else "",
                            "start_line": i + 1 - len(current_code),
                            "end_line": i + 1,
                        }
                    )
                    current_code = []
                    in_code_block = False
                else:
                    in_code_block = True
                    current_code = []
            elif in_code_block:
                current_code.append(line)

        # リンク抽出
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        links = re.findall(link_pattern, content)
        structure["links"] = [{"text": text, "url": url} for text, url in links]

        return structure

    @staticmethod
    def extract_html_structure(content: str) -> dict[str, Any]:
        """HTML構造を抽出"""
        structure = {
            "elements": [],
            "headings": [],
            "lists": [],
            "tables": [],
        }

        # 見出し抽出
        heading_pattern = r"<h([1-6])[^>]*>([^<]+)</h[1-6]>"
        headings = re.findall(heading_pattern, content, re.IGNORECASE)

        for level, text in headings:
            structure["headings"].append(
                {
                    "level": int(level),
                    "text": text.strip(),
                }
            )

        # リスト抽出
        list_pattern = r"<(ul|ol)[^>]*>(.*?)</\1>"
        lists = re.findall(list_pattern, content, re.DOTALL | re.IGNORECASE)

        for list_type, list_content in lists:
            item_pattern = r"<li[^>]*>([^<]+)</li>"
            items = re.findall(item_pattern, list_content, re.IGNORECASE)
            structure["lists"].append(
                {
                    "type": list_type.lower(),
                    "items": [item.strip() for item in items],
                }
            )

        # テーブル抽出
        table_pattern = r"<table[^>]*>(.*?)</table>"
        tables = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)

        for table_content in tables:
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = re.findall(row_pattern, table_content, re.DOTALL | re.IGNORECASE)

            table_data = []
            for row in rows:
                cell_pattern = r"<t[hd][^>]*>([^<]+)</t[hd]>"
                cells = re.findall(cell_pattern, row, re.IGNORECASE)
                if cells:
                    table_data.append([cell.strip() for cell in cells])

            if table_data:
                structure["tables"].append(
                    {
                        "rows": table_data,
                        "row_count": len(table_data),
                        "column_count": len(table_data[0]) if table_data else 0,
                    }
                )

        # 一般的な要素
        structure["elements"] = []
        if structure["lists"]:
            structure["elements"].extend(
                [{"type": "list", "count": len(structure["lists"])}]
            )
        if structure["tables"]:
            structure["elements"].extend(
                [{"type": "table", "count": len(structure["tables"])}]
            )

        return structure


class EntityExtractor:
    """エンティティ抽出器"""

    @staticmethod
    def extract_entities(
        content: str, confidence_threshold: float = 0.5
    ) -> dict[str, Any]:
        """エンティティを抽出"""
        entities = {
            "technical_terms": [],
            "http_codes": [],
            "email_addresses": [],
            "urls": [],
            "file_paths": [],
            "version_numbers": [],
        }

        # 技術用語（大文字で始まる略語、API関連用語など）
        tech_terms = re.findall(
            r"\b[A-Z]{2,}\b|API|HTTP|JSON|XML|SQL|REST|SOAP", content
        )
        entities["technical_terms"] = list(set(tech_terms))

        # HTTPステータスコード
        http_codes = re.findall(r"\b[1-5]\d{2}\b", content)
        entities["http_codes"] = list(set(http_codes))

        # メールアドレス
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, content)
        entities["email_addresses"] = list(set(emails))

        # URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        entities["urls"] = list(set(urls))

        # ファイルパス
        file_path_pattern = r"[./~][/\w.-]+\.[a-zA-Z]{2,4}"
        file_paths = re.findall(file_path_pattern, content)
        entities["file_paths"] = list(set(file_paths))

        # バージョン番号
        version_pattern = r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b"
        versions = re.findall(version_pattern, content)
        entities["version_numbers"] = list(set(versions))

        return entities


class KeywordExtractor:
    """キーワード抽出器"""

    @staticmethod
    def extract_keywords(
        content: str, max_keywords: int = 20, language: str = "en"
    ) -> list[dict[str, Any]]:
        """キーワードを抽出"""
        # ストップワード（簡易版）
        stop_words_en = {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "so",
            "up",
            "out",
            "if",
            "about",
            "who",
            "get",
            "which",
            "go",
            "me",
            "when",
            "make",
            "can",
            "like",
            "time",
            "no",
            "just",
            "him",
            "know",
            "take",
            "people",
            "into",
            "year",
            "your",
            "good",
            "some",
            "could",
            "them",
            "see",
            "other",
            "than",
            "then",
            "now",
            "look",
            "only",
            "come",
            "its",
            "over",
            "think",
            "also",
            "back",
            "after",
            "use",
            "two",
            "how",
            "our",
            "work",
            "first",
            "well",
            "way",
            "even",
            "new",
            "want",
            "because",
            "any",
            "these",
            "give",
            "day",
            "most",
            "us",
        }

        stop_words_ja = {
            "の",
            "に",
            "は",
            "を",
            "た",
            "が",
            "で",
            "て",
            "と",
            "し",
            "れ",
            "さ",
            "ある",
            "いる",
            "も",
            "する",
            "から",
            "な",
            "こと",
            "として",
            "い",
            "や",
            "れる",
            "など",
            "なっ",
            "ない",
            "この",
            "ため",
            "その",
            "あっ",
            "よう",
            "また",
            "もの",
            "という",
            "あり",
            "まで",
            "られ",
            "なる",
            "へ",
            "か",
            "だ",
            "これ",
            "によって",
            "により",
            "おり",
            "より",
            "による",
            "ず",
            "なり",
            "られる",
            "において",
            "ば",
            "なかっ",
            "なく",
            "しかし",
            "について",
            "せ",
            "だっ",
            "その後",
            "方",
            "今",
            "前",
            "後",
            "間",
            "上",
            "下",
            "中",
            "内",
            "外",
            "左",
            "右",
            "先",
            "奥",
            "手",
            "本",
            "事",
            "物",
            "人",
            "者",
            "時",
            "日",
            "年",
            "月",
            "週",
            "国",
            "家",
            "会社",
        }

        stop_words = stop_words_ja if language == "ja" else stop_words_en

        # テキストを単語に分割
        words = re.findall(r"\b\w+\b", content.lower())

        # ストップワードを除去
        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        # 単語の頻度を計算
        word_counts = Counter(filtered_words)

        # 最も頻繁な単語を取得
        top_words = word_counts.most_common(max_keywords)

        # キーワードオブジェクトを作成
        keywords = []
        total_words = len(filtered_words)

        for word, count in top_words:
            frequency = count / total_words if total_words > 0 else 0
            confidence = min(frequency * 10, 1.0)  # 簡易的な信頼度計算

            keywords.append(
                {
                    "text": word,
                    "frequency": count,
                    "confidence": confidence,
                    "relative_frequency": frequency,
                }
            )

        return keywords


class StatisticsExtractor:
    """統計情報抽出器"""

    @staticmethod
    def extract_statistics(
        content: str, structure: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """統計情報を抽出"""
        lines = content.split("\n")
        sentences = re.split(r"[.!?。！？]", content)
        words = re.findall(r"\b\w+\b", content)
        paragraphs = re.split(r"\n\s*\n", content)

        # 基本統計
        basic_stats = {
            "char_count": len(content),
            "word_count": len(words),
            "line_count": len(lines),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
        }

        # 複雑度指標
        avg_sentence_length = basic_stats["word_count"] / max(
            basic_stats["sentence_count"], 1
        )
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

        # 簡易的な読みやすさスコア（Flesch Reading Ease の簡易版）
        readability_score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
        )
        readability_score = max(0, min(100, readability_score))

        complexity = {
            "readability_score": round(readability_score, 2),
            "sentence_length_avg": round(avg_sentence_length, 2),
            "word_length_avg": round(avg_word_length, 2),
        }

        # 構造統計
        structure_stats = {
            "heading_count": 0,
            "list_count": 0,
            "code_block_count": 0,
            "link_count": 0,
        }

        if structure:
            structure_stats["heading_count"] = len(structure.get("headings", []))
            structure_stats["list_count"] = len(structure.get("lists", []))
            structure_stats["code_block_count"] = len(structure.get("code_blocks", []))
            structure_stats["link_count"] = len(structure.get("links", []))

        return {
            **basic_stats,
            "complexity": complexity,
            "structure_stats": structure_stats,
        }


class MetadataExtractor:
    """メタデータ抽出器メインクラス"""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.language_detector = LanguageDetector()
        self.structure_extractor = StructureExtractor()
        self.entity_extractor = EntityExtractor()
        self.keyword_extractor = KeywordExtractor()
        self.statistics_extractor = StatisticsExtractor()

    async def extract_metadata(self, document: dict[str, Any]) -> ExtractionResult:
        """ドキュメントからメタデータを抽出"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            content = document.get("content", "")
            if not content.strip():
                return ExtractionResult(
                    success=False,
                    document_id=document.get("id", "unknown"),
                    metadata={},
                    field_types={},
                    processing_time=0.0,
                    error_message="Document content is empty",
                )

            document_id = document.get("id", "unknown")
            metadata = {}
            field_types = {}

            # ドキュメントタイプ検出
            doc_type = self._detect_document_type(document)
            metadata["document_type"] = doc_type
            field_types["document_type"] = FieldType.STRING

            # 言語検出
            if self.config.language_detection:
                language, confidence = self.language_detector.detect_language(content)
                metadata["language"] = {
                    "detected": language,
                    "confidence": confidence,
                }
                field_types["language"] = FieldType.OBJECT
            else:
                language = "en"  # デフォルト

            # 構造抽出
            structure = None
            if self.config.extract_structure:
                if doc_type == DocumentType.MARKDOWN:
                    structure = self.structure_extractor.extract_markdown_structure(
                        content
                    )
                elif doc_type == DocumentType.HTML:
                    structure = self.structure_extractor.extract_html_structure(content)

                if structure:
                    metadata["structure"] = structure
                    field_types["structure"] = FieldType.OBJECT

            # エンティティ抽出
            if self.config.extract_entities:
                entities = self.entity_extractor.extract_entities(
                    content, self.config.confidence_threshold
                )
                metadata["entities"] = entities
                field_types["entities"] = FieldType.OBJECT

            # キーワード抽出
            if self.config.extract_keywords:
                keywords = self.keyword_extractor.extract_keywords(
                    content, self.config.max_keywords, language
                )
                # 信頼度でフィルタリング
                filtered_keywords = [
                    kw
                    for kw in keywords
                    if kw["confidence"] >= self.config.confidence_threshold
                ]
                metadata["keywords"] = filtered_keywords
                field_types["keywords"] = FieldType.ARRAY

            # 統計情報抽出
            if self.config.extract_statistics:
                statistics = self.statistics_extractor.extract_statistics(
                    content, structure
                )
                metadata["statistics"] = statistics
                field_types["statistics"] = FieldType.OBJECT

            # 既存メタデータの保持
            if "existing_metadata" in document:
                metadata["existing_metadata"] = document["existing_metadata"]
                field_types["existing_metadata"] = FieldType.OBJECT

            # 抽出時刻の追加
            metadata["extraction_info"] = {
                "extracted_at": datetime.now().isoformat(),
                "extractor_version": "1.0.0",
                "config": {
                    "structure": self.config.extract_structure,
                    "entities": self.config.extract_entities,
                    "keywords": self.config.extract_keywords,
                    "statistics": self.config.extract_statistics,
                },
            }
            field_types["extraction_info"] = FieldType.OBJECT

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ExtractionResult(
                success=True,
                document_id=document_id,
                metadata=metadata,
                field_types=field_types,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ExtractionResult(
                success=False,
                document_id=document.get("id", "unknown"),
                metadata={},
                field_types={},
                processing_time=processing_time,
                error_message=str(e),
            )

    def _detect_document_type(self, document: dict[str, Any]) -> DocumentType:
        """ドキュメントタイプを検出"""
        content = document.get("content", "")
        file_type = document.get("file_type", "").lower()
        source_type = document.get("source_type", "").lower()

        # ソースタイプから判定
        if source_type == "confluence":
            return DocumentType.CONFLUENCE
        elif source_type == "jira":
            return DocumentType.JIRA

        # ファイル拡張子から判定
        if file_type in ["md", "markdown"]:
            return DocumentType.MARKDOWN
        elif file_type in ["html", "htm"]:
            return DocumentType.HTML
        elif file_type == "json":
            return DocumentType.JSON
        elif file_type == "xml":
            return DocumentType.XML
        elif file_type == "pdf":
            return DocumentType.PDF

        # コンテンツから判定
        if re.search(r"<[^>]+>", content):
            if re.search(r"<h[1-6]|<p>|<div>", content, re.IGNORECASE):
                return DocumentType.HTML
        elif re.search(r"^#{1,6}\s", content, re.MULTILINE):
            return DocumentType.MARKDOWN
        elif content.strip().startswith("{") and content.strip().endswith("}"):
            return DocumentType.JSON
        elif content.strip().startswith("<") and content.strip().endswith(">"):
            return DocumentType.XML

        return DocumentType.PLAIN_TEXT
