"""ドキュメントチャンクリポジトリ"""

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DocumentChunk


class DocumentChunkRepository:
    """ドキュメントチャンクのCRUD操作を管理するリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, chunk: DocumentChunk) -> DocumentChunk:
        """チャンクを作成"""
        self.session.add(chunk)
        await self.session.commit()
        await self.session.refresh(chunk)
        return chunk

    async def get_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """IDでチャンクを取得"""
        result = await self.session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.id == chunk_id)
        )
        return result.scalar_one_or_none()  # type: ignore

    async def get_by_document_id(self, document_id: str) -> list[DocumentChunk]:
        """ドキュメントIDでチャンクを取得"""
        result = await self.session.execute(
            sa.select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_type(self, chunk_type: str) -> list[DocumentChunk]:
        """チャンクタイプで取得"""
        result = await self.session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.chunk_type == chunk_type)
        )
        return list(result.scalars().all())

    async def update(self, chunk: DocumentChunk) -> DocumentChunk:
        """チャンクを更新"""
        await self.session.commit()
        await self.session.refresh(chunk)
        return chunk

    async def delete(self, chunk_id: str) -> bool:
        """チャンクを削除"""
        result = await self.session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.id == chunk_id)
        )
        chunk = result.scalar_one_or_none()

        if chunk:
            await self.session.delete(chunk)
            await self.session.commit()
            return True
        return False

    async def delete_by_document_id(self, document_id: str) -> int:
        """ドキュメントIDでチャンクを削除"""
        result = await self.session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()

        count = 0
        for chunk in chunks:
            await self.session.delete(chunk)
            count += 1

        await self.session.commit()
        return count

    async def search_by_content(self, search_term: str) -> list[DocumentChunk]:
        """コンテンツでの検索"""
        query = sa.select(DocumentChunk).where(
            sa.or_(
                DocumentChunk.title.contains(search_term),
                DocumentChunk.content.contains(search_term),
            )
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_chunks_by_size_range(
        self, min_size: int, max_size: int
    ) -> list[DocumentChunk]:
        """サイズ範囲でチャンクを取得"""
        query = sa.select(DocumentChunk).where(
            DocumentChunk.content_length >= min_size,
            DocumentChunk.content_length <= max_size,
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())
