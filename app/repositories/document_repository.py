"""ドキュメントリポジトリ"""

from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database import Document, DocumentChunk


class DocumentRepository:
    """ドキュメントのCRUD操作を管理するリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, document: Document) -> Document:
        """ドキュメントを作成"""
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """IDでドキュメントを取得"""
        result = await self.session.execute(
            sa.select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_source(
        self,
        source_type: str,
        source_id: str
    ) -> Optional[Document]:
        """ソースタイプとIDでドキュメントを取得"""
        result = await self.session.execute(
            sa.select(Document).where(
                Document.source_type == source_type,
                Document.source_id == source_id
            )
        )
        return result.scalar_one_or_none()

    async def get_with_chunks(self, document_id: str) -> Optional[Document]:
        """チャンクも含めてドキュメントを取得"""
        result = await self.session.execute(
            sa.select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def update(self, document: Document) -> Document:
        """ドキュメントを更新"""
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def delete(self, document_id: str) -> bool:
        """ドキュメントを削除"""
        result = await self.session.execute(
            sa.select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if document:
            await self.session.delete(document)
            await self.session.commit()
            return True
        return False

    async def list_documents(
        self,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Document]:
        """ドキュメント一覧を取得"""
        query = sa.select(Document)

        # フィルタ条件
        if source_type:
            query = query.where(Document.source_type == source_type)
        if status:
            query = query.where(Document.status == status)

        # ページネーション
        query = query.limit(limit).offset(offset)

        # 作成日時順でソート
        query = query.order_by(Document.created_at.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def search_by_content(self, search_term: str) -> List[Document]:
        """コンテンツでの検索"""
        query = sa.select(Document).where(
            sa.or_(
                Document.title.contains(search_term),
                Document.content.contains(search_term)
            )
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_by_source_type(self, source_type: str) -> int:
        """ソースタイプ別のドキュメント数を取得"""
        result = await self.session.execute(
            sa.select(sa.func.count(Document.id)).where(
                Document.source_type == source_type
            )
        )
        return result.scalar() or 0

    async def get_outdated_documents(
        self,
        hours: int = 24
    ) -> List[Document]:
        """指定時間以上更新されていないドキュメントを取得"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        query = sa.select(Document).where(
            Document.updated_at < cutoff_time,
            Document.status == "active"
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())
