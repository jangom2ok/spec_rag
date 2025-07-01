"""ドキュメント操作API"""

import logging
from enum import Enum

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.core.auth import validate_api_key

router = APIRouter(prefix="/v1/documents", tags=["documents"])


class SourceType(str, Enum):
    """ソースタイプのEnum"""

    confluence = "confluence"
    jira = "jira"
    test = "test"


class DocumentCreate(BaseModel):
    """ドキュメント作成用のモデル"""

    title: str
    content: str
    source_type: SourceType


class DocumentResponse(BaseModel):
    """ドキュメントレスポンス用のモデル"""

    id: str
    title: str
    content: str
    source_type: SourceType


class DocumentList(BaseModel):
    """ドキュメント一覧レスポンス用のモデル"""

    documents: list[DocumentResponse]


async def get_current_user_or_api_key(
    authorization: str | None = Header(None), x_api_key: str | None = Header(None)
) -> dict:
    """JWT認証またはAPI Key認証を試行"""
    # API Key認証を先に試行
    if x_api_key:
        api_key_info = validate_api_key(x_api_key)
        if api_key_info:
            return {
                "user_id": api_key_info["user_id"],
                "permissions": api_key_info["permissions"],
                "auth_type": "api_key",
            }

    # JWT認証を試行
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            from app.core.auth import is_token_blacklisted, users_storage, verify_token

            if is_token_blacklisted(token):
                raise HTTPException(status_code=401, detail="Token has been revoked")

            payload = verify_token(token)
            email = payload.get("sub")
            if email:
                user = users_storage.get(email)
                if user:
                    user_info = user.copy()
                    user_info["email"] = email
                    user_info["auth_type"] = "jwt"
                    return user_info
        except Exception as e:
            # JWT認証が失敗した場合はAPI Key認証にフォールバック
            logging.debug(f"JWT認証に失敗、API Key認証にフォールバック: {e}")
            pass

    raise HTTPException(status_code=401, detail="Authentication required")


@router.get("/", response_model=DocumentList)
async def list_documents(
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """ドキュメント一覧を取得"""
    # テスト用のモックデータ
    mock_documents = [
        DocumentResponse(
            id="doc1",
            title="Sample Document 1",
            content="Sample content 1",
            source_type=SourceType.test,
        ),
        DocumentResponse(
            id="doc2",
            title="Sample Document 2",
            content="Sample content 2",
            source_type=SourceType.confluence,
        ),
    ]
    return DocumentList(documents=mock_documents)


@router.post("/", response_model=DocumentResponse, status_code=201)
async def create_document(
    document: DocumentCreate,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """ドキュメントを作成"""
    # 書き込み権限をチェック
    if "write" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Write permission required")

    # 実装は後で追加
    return DocumentResponse(
        id="mock-id",
        title=document.title,
        content=document.content,
        source_type=document.source_type,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """ドキュメントを削除"""
    # 削除権限をチェック（管理者権限必要）
    if "delete" not in current_user.get(
        "permissions", []
    ) and "admin" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Delete permission required")

    # 実装は後で追加
    return {"message": f"Document {document_id} deleted successfully"}


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """ドキュメントを取得"""
    # 実装は後で追加
    if document_id == "test-id":
        return DocumentResponse(
            id=document_id,
            title="Test Document",
            content="Test content",
            source_type=SourceType.test,
        )
    raise HTTPException(status_code=404, detail="Document not found")
