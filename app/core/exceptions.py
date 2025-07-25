"""カスタム例外クラス"""


class RAGSystemError(Exception):
    """RAGシステム基底例外クラス"""

    def __init__(
        self, message: str, error_code: str = "RAG_SYSTEM_ERROR", status_code: int = 500
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class DatabaseError(RAGSystemError):
    """データベース関連の例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="DATABASE_ERROR", status_code=500)


class VectorDatabaseError(RAGSystemError):
    """ベクトルデータベース関連の例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="VECTOR_DATABASE_ERROR", status_code=500)


class ValidationError(RAGSystemError):
    """バリデーション例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="VALIDATION_ERROR", status_code=422)


class AuthenticationError(RAGSystemError):
    """認証例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", status_code=401)


class AuthorizationError(RAGSystemError):
    """認可例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="AUTHORIZATION_ERROR", status_code=403)
