from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_addition_basic_equals_two():
    """基本的な加算操作が正しく動作することを確認するテスト。

    期待される結果:
        1 + 1 が 2 と等しいことを確認します。
    """
    assert 1 + 1 == 2


def test_read_root():
    """ルートエンドポイントが正しく動作することを確認するテスト。

    期待される結果:
        - ステータスコード200が返される
        - レスポンスボディに"Hello, World!"メッセージが含まれる
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}
