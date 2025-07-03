"""ドキュメントチャンク化サービス

TDD実装：テストケースに基づいたドキュメントチャンク化機能
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """チャンク化戦略"""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class ChunkType(str, Enum):
    """チャンクタイプ"""

    TEXT = "text"
    HEADING = "heading"
    SECTION = "section"
    LIST = "list"
    TABLE = "table"


@dataclass
class ChunkingConfig:
    """チャンク化設定"""

    strategy: ChunkingStrategy
    chunk_size: int = 1000
    overlap_size: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    preserve_sections: bool = False
    preserve_headings: bool = False
    similarity_threshold: float = 0.8
    max_hierarchy_depth: int = 5
    deduplicate_chunks: bool = False

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.overlap_size < 0:
            raise ValueError("overlap_size must be non-negative")
        if self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than 0")


@dataclass
class DocumentChunk:
    """ドキュメントチャンク"""

    id: str
    document_id: str
    chunk_index: int
    chunk_type: ChunkType
    content: str
    content_length: int
    token_count: int
    title: str | None = None
    hierarchy_path: str | None = None
    chunk_metadata: dict[str, Any] | None = None
    language: str = "en"

    def __post_init__(self):
        """チャンク作成後の処理"""
        if not self.content_length:
            self.content_length = len(self.content)
        if not self.token_count:
            self.token_count = self._estimate_token_count()

    def _estimate_token_count(self) -> int:
        """トークン数の推定（簡易的な実装）"""
        # 単語数ベースの簡易推定
        words = self.content.split()
        return len(words)


@dataclass
class ChunkResult:
    """チャンク化結果"""

    success: bool
    chunks: list[DocumentChunk]
    total_chunks: int
    processing_time: float
    error_message: str | None = None

    def get_summary(self) -> dict[str, Any]:
        """チャンク結果のサマリーを取得"""
        total_content_length = sum(chunk.content_length for chunk in self.chunks)
        average_chunk_size = (
            total_content_length / len(self.chunks) if self.chunks else 0
        )

        return {
            "total_chunks": self.total_chunks,
            "average_chunk_size": average_chunk_size,
            "total_content_length": total_content_length,
            "processing_time": self.processing_time,
            "success": self.success,
        }


class TextSplitter:
    """テキスト分割ユーティリティ"""

    @staticmethod
    def split_by_sentences(text: str, language: str = "en") -> list[str]:
        """文単位でテキストを分割"""
        if language == "ja":
            # 日本語の文区切り（改行も含める）
            sentences = re.split(r"([。！？])", text)
        else:
            # 英語の文区切り（改行も含める）
            sentences = re.split(r"([.!?])", text)

        # 区切り文字と文を組み合わせる
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                combined.append(sentence + punct)

        # 最後の文が区切り文字なしの場合
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            combined.append(sentences[-1].strip())

        return [sent.strip() for sent in combined if sent.strip()]

    @staticmethod
    def split_by_paragraphs(text: str) -> list[str]:
        """段落単位でテキストを分割"""
        paragraphs = re.split(r"\n\s*\n", text)
        return [para.strip() for para in paragraphs if para.strip()]

    @staticmethod
    def detect_headings(text: str) -> list[tuple[str, int, str]]:
        """見出しを検出（Markdown形式）"""
        headings = []
        lines = text.split("\n")

        for _i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            if line.startswith("#") and " " in line:
                # #の数をカウント
                hash_count = 0
                for char in line:
                    if char == "#":
                        hash_count += 1
                    else:
                        break

                # タイトルを抽出
                title = line[hash_count:].strip()
                if title:
                    headings.append((title, hash_count, original_line.strip()))

        return headings

    @staticmethod
    def detect_language(text: str) -> str:
        """簡易的な言語検出"""
        # 日本語文字の検出
        japanese_chars = len(
            re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text)
        )
        total_chars = len(re.findall(r"[^\s]", text))

        if total_chars > 0 and japanese_chars / total_chars > 0.1:
            return "ja"
        return "en"


class FixedSizeChunker:
    """固定サイズチャンカー"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.text_splitter = TextSplitter()

    async def chunk(self, document: dict[str, Any]) -> list[DocumentChunk]:
        """固定サイズでチャンク化"""
        content = document["content"]
        document_id = document["id"]
        language = self.text_splitter.detect_language(content)

        chunks = []
        chunk_index = 0

        if self.config.preserve_paragraphs:
            paragraphs = self.text_splitter.split_by_paragraphs(content)
            current_chunk = ""

            for paragraph in paragraphs:
                # パラグラフを追加できるかチェック
                test_chunk = (
                    current_chunk + ("\n\n" if current_chunk else "") + paragraph
                )

                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # 現在のチャンクを保存
                    if current_chunk.strip():
                        chunks.append(
                            self._create_chunk(
                                document_id,
                                chunk_index,
                                current_chunk.strip(),
                                language,
                            )
                        )
                        chunk_index += 1

                    # 新しいチャンクを開始
                    # オーバーラップ処理
                    if self.config.overlap_size > 0 and current_chunk:
                        overlap_text = current_chunk[
                            -self.config.overlap_size :
                        ].strip()
                        if overlap_text:
                            current_chunk = overlap_text + "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        current_chunk = paragraph

                    # 新しいチャンクも大きすぎる場合は強制分割
                    if len(current_chunk) > self.config.chunk_size:
                        chunk_content = current_chunk[: self.config.chunk_size]
                        chunks.append(
                            self._create_chunk(
                                document_id, chunk_index, chunk_content, language
                            )
                        )
                        chunk_index += 1
                        current_chunk = current_chunk[
                            self.config.chunk_size - self.config.overlap_size :
                        ]

            if current_chunk.strip():
                chunks.append(
                    self._create_chunk(
                        document_id, chunk_index, current_chunk.strip(), language
                    )
                )

        elif self.config.preserve_sentences:
            sentences = self.text_splitter.split_by_sentences(content, language)
            current_chunk = ""

            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue

                # 句読点を追加
                if not sentence_clean.endswith((".", "。", "!", "?", "！", "？")):
                    sentence_clean += "。" if language == "ja" else "."

                # 文を追加できるかチェック
                test_chunk = (
                    current_chunk + (" " if current_chunk else "") + sentence_clean
                )

                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # 現在のチャンクを保存
                    if current_chunk.strip():
                        chunks.append(
                            self._create_chunk(
                                document_id,
                                chunk_index,
                                current_chunk.strip(),
                                language,
                            )
                        )
                        chunk_index += 1

                    # 新しいチャンクを開始
                    if self.config.overlap_size > 0 and current_chunk:
                        overlap_text = current_chunk[
                            -self.config.overlap_size :
                        ].strip()
                        if overlap_text:
                            current_chunk = overlap_text + " " + sentence_clean
                        else:
                            current_chunk = sentence_clean
                    else:
                        current_chunk = sentence_clean

                    # 新しいチャンクも大きすぎる場合は強制分割
                    if len(current_chunk) > self.config.chunk_size:
                        chunk_content = current_chunk[: self.config.chunk_size]
                        chunks.append(
                            self._create_chunk(
                                document_id, chunk_index, chunk_content, language
                            )
                        )
                        chunk_index += 1
                        current_chunk = current_chunk[
                            self.config.chunk_size - self.config.overlap_size :
                        ]

            if current_chunk.strip():
                chunks.append(
                    self._create_chunk(
                        document_id, chunk_index, current_chunk.strip(), language
                    )
                )

        else:
            # 単純な固定サイズ分割
            step_size = max(1, self.config.chunk_size - self.config.overlap_size)
            for i in range(0, len(content), step_size):
                chunk_content = content[i : i + self.config.chunk_size]
                if len(
                    chunk_content
                ) >= self.config.min_chunk_size or i + self.config.chunk_size >= len(
                    content
                ):
                    chunks.append(
                        self._create_chunk(
                            document_id, chunk_index, chunk_content, language
                        )
                    )
                    chunk_index += 1

        return chunks

    def _create_chunk(
        self, document_id: str, chunk_index: int, content: str, language: str
    ) -> DocumentChunk:
        """チャンクを作成"""
        chunk_id = str(uuid.uuid4())

        # メタデータの生成
        metadata = {
            "position": {
                "start": chunk_index * self.config.chunk_size,
                "end": chunk_index * self.config.chunk_size + len(content),
            },
            "relative_position": chunk_index,
        }

        return DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_type=ChunkType.TEXT,
            content=content,
            content_length=len(content),
            token_count=len(content.split()),
            chunk_metadata=metadata,
            language=language,
        )


class SemanticChunker:
    """セマンティックチャンカー"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.text_splitter = TextSplitter()

    async def chunk(self, document: dict[str, Any]) -> list[DocumentChunk]:
        """セマンティック境界でチャンク化"""
        content = document["content"]
        document_id = document["id"]
        language = self.text_splitter.detect_language(content)

        chunks = []
        chunk_index = 0

        # セクション単位での分割を試行
        if self.config.preserve_sections:
            sections = self._extract_sections(content)

            for section_title, section_content in sections:
                if len(section_content) > self.config.chunk_size:
                    # 大きなセクションはさらに分割
                    sub_chunks = await self._split_large_section(
                        section_content,
                        document_id,
                        chunk_index,
                        language,
                        section_title,
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    chunk = self._create_semantic_chunk(
                        document_id,
                        chunk_index,
                        section_content,
                        language,
                        section_title,
                        ChunkType.SECTION,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        else:
            # 段落ベースのセマンティック分割
            paragraphs = self.text_splitter.split_by_paragraphs(content)
            current_chunk = ""

            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= self.config.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunk = self._create_semantic_chunk(
                            document_id, chunk_index, current_chunk.strip(), language
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = paragraph + "\n\n"

            if current_chunk.strip():
                chunk = self._create_semantic_chunk(
                    document_id, chunk_index, current_chunk.strip(), language
                )
                chunks.append(chunk)

        return chunks

    def _extract_sections(self, content: str) -> list[tuple[str, str]]:
        """セクションを抽出"""
        sections = []
        lines = content.split("\n")
        current_section_title = ""
        current_section_content = []

        for line in lines:
            if line.strip().startswith("#"):
                # 前のセクションを保存
                if current_section_content:
                    sections.append(
                        (current_section_title, "\n".join(current_section_content))
                    )

                # 新しいセクション開始
                current_section_title = line.strip()
                current_section_content = [line]
            else:
                current_section_content.append(line)

        # 最後のセクションを保存
        if current_section_content:
            sections.append((current_section_title, "\n".join(current_section_content)))

        return sections

    async def _split_large_section(
        self,
        content: str,
        document_id: str,
        start_index: int,
        language: str,
        section_title: str,
    ) -> list[DocumentChunk]:
        """大きなセクションを分割"""
        # 固定サイズチャンカーを使用して分割
        fixed_chunker = FixedSizeChunker(self.config)
        temp_doc = {"id": document_id, "content": content}

        sub_chunks = await fixed_chunker.chunk(temp_doc)

        # チャンクインデックスと階層パスを調整
        for i, chunk in enumerate(sub_chunks):
            chunk.chunk_index = start_index + i
            chunk.hierarchy_path = section_title
            chunk.chunk_type = ChunkType.SECTION if i == 0 else ChunkType.TEXT

        return sub_chunks

    def _create_semantic_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        language: str,
        title: str | None = None,
        chunk_type: ChunkType = ChunkType.TEXT,
    ) -> DocumentChunk:
        """セマンティックチャンクを作成"""
        chunk_id = str(uuid.uuid4())

        metadata = {
            "semantic_boundary": True,
            "section_title": title,
            "position": {"index": chunk_index},
        }

        return DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            title=title,
            content=content,
            content_length=len(content),
            token_count=len(content.split()),
            chunk_metadata=metadata,
            language=language,
        )


class HierarchicalChunker:
    """階層チャンカー"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.text_splitter = TextSplitter()

    async def chunk(self, document: dict[str, Any]) -> list[DocumentChunk]:
        """階層構造を保持してチャンク化"""
        content = document["content"]
        document_id = document["id"]
        language = self.text_splitter.detect_language(content)

        # 見出し構造を抽出
        headings = self.text_splitter.detect_headings(content)
        hierarchy = self._build_hierarchy(headings, content)

        chunks = []
        chunk_index = 0

        for section in hierarchy:
            section_chunks = await self._chunk_section(
                section, document_id, chunk_index, language
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _build_hierarchy(
        self, headings: list[tuple[str, int, str]], content: str
    ) -> list[dict[str, Any]]:
        """階層構造を構築"""
        if not headings:
            return [
                {
                    "title": "",
                    "level": 0,
                    "content": content,
                    "path": "",
                    "children": [],
                }
            ]

        hierarchy = []
        lines = content.split("\n")
        _current_section = None

        for i, (title, level, heading_line) in enumerate(headings):
            # セクションの開始位置を見つける
            start_idx = next(
                (j for j, line in enumerate(lines) if line.strip() == heading_line), 0
            )

            # セクションの終了位置を見つける
            end_idx = len(lines)
            if i + 1 < len(headings):
                next_heading_line = headings[i + 1][2]
                end_idx = next(
                    (
                        j
                        for j, line in enumerate(lines)
                        if line.strip() == next_heading_line
                    ),
                    len(lines),
                )

            section_content = "\n".join(lines[start_idx:end_idx])

            section = {
                "title": title,
                "level": level,
                "content": section_content,
                "path": self._build_path(hierarchy, title, level),
                "children": [],
            }

            hierarchy.append(section)

        return hierarchy

    def _build_path(
        self, existing_sections: list[dict[str, Any]], title: str, level: int
    ) -> str:
        """階層パスを構築"""
        path_parts = []

        # 上位レベルのセクションを探す
        for section in reversed(existing_sections):
            if section["level"] < level:
                if section["path"]:
                    path_parts = section["path"].split("/") + [section["title"]]
                else:
                    path_parts = [section["title"]]
                break

        path_parts.append(title)
        return "/".join(path_parts)

    async def _chunk_section(
        self, section: dict[str, Any], document_id: str, start_index: int, language: str
    ) -> list[DocumentChunk]:
        """セクションをチャンク化"""
        content = section["content"]
        title = section["title"]
        path = section["path"]

        chunks = []

        # ヘッディングチャンクを作成
        if title and self.config.preserve_headings:
            heading_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=start_index,
                chunk_type=ChunkType.HEADING,
                title=title,
                content=title,
                content_length=len(title),
                token_count=len(title.split()),
                hierarchy_path=path,
                language=language,
            )
            chunks.append(heading_chunk)
            start_index += 1

        # コンテンツをチャンク化
        if len(content) > self.config.chunk_size:
            # 大きなセクションは固定サイズで分割
            fixed_chunker = FixedSizeChunker(self.config)
            temp_doc = {"id": document_id, "content": content}

            content_chunks = await fixed_chunker.chunk(temp_doc)

            for i, chunk in enumerate(content_chunks):
                chunk.chunk_index = start_index + i
                chunk.hierarchy_path = path
                chunk.title = title if i == 0 else None
                chunks.append(chunk)
        else:
            # 小さなセクションはそのまま1つのチャンクに
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=start_index,
                chunk_type=ChunkType.SECTION,
                title=title,
                content=content,
                content_length=len(content),
                token_count=len(content.split()),
                hierarchy_path=path,
                language=language,
            )
            chunks.append(chunk)

        return chunks


class DocumentChunker:
    """ドキュメントチャンカーメインクラス"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(config),
            ChunkingStrategy.SEMANTIC: SemanticChunker(config),
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunker(config),
        }

    async def chunk_document(self, document: dict[str, Any]) -> ChunkResult:
        """ドキュメントをチャンク化"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            if not document.get("content"):
                return ChunkResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    processing_time=0.0,
                    error_message="Document content is empty",
                )

            # チャンク化実行
            chunker = self._chunkers[self.config.strategy]
            chunks = await chunker.chunk(document)

            # 重複排除
            if self.config.deduplicate_chunks:
                chunks = self._deduplicate_chunks(chunks)

            # 結果の作成
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ChunkResult(
                success=True,
                chunks=chunks,
                total_chunks=len(chunks),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ChunkResult(
                success=False,
                chunks=[],
                total_chunks=0,
                processing_time=processing_time,
                error_message=str(e),
            )

    def _deduplicate_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """チャンクの重複を排除"""
        seen_contents: set[str] = set()
        deduplicated = []

        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append(chunk)

        # インデックスを再割り当て
        for i, chunk in enumerate(deduplicated):
            chunk.chunk_index = i

        return deduplicated
