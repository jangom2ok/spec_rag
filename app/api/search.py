"""検索API"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    """検索リクエスト用のモデル"""

    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    """検索結果用のモデル"""

    id: str
    title: str
    content: str
    score: float


class SearchResponse(BaseModel):
    """検索レスポンス用のモデル"""

    results: list[SearchResult]
    total: int


@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """ドキュメント検索"""
    # 実装は後で追加
    mock_results = [
        SearchResult(
            id="mock-1", title="Mock Document 1", content="Mock content 1", score=0.9
        ),
        SearchResult(
            id="mock-2", title="Mock Document 2", content="Mock content 2", score=0.8
        ),
    ]

    return SearchResponse(
        results=mock_results[: request.top_k], total=len(mock_results)
    )
