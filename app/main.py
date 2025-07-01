from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    """ルートエンドポイントのハンドラー。

    Returns:
        dict[str, str]: 'Hello, World!'メッセージを含む辞書。
    """
    return {"message": "Hello, World!"}
