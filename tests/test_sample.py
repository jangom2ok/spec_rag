from fastapi.testclient import TestClient
import pytest

from app.main import app


@pytest.fixture
def client():
    """FastAPI TestClientのfixture。"""
    return TestClient(app)


def test_read_root(client):
    """ルートエンドポイントが正しく動作することを確認するテスト。

    期待される結果:
        - ステータスコード200が返される
        - レスポンスボディに"Hello, World!"メッセージが含まれる
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}
