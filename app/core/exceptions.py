"""カスタム例外クラス"""


class RAGSystemException(Exception):
    """RAGシステム基底例外クラス"""

    def __init__(
        self, message: str, error_code: str = "RAG_SYSTEM_ERROR", status_code: int = 500
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class DatabaseException(RAGSystemException):
    """データベース関連の例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="DATABASE_ERROR", status_code=500)


class VectorDatabaseException(RAGSystemException):
    """ベクトルデータベース関連の例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="VECTOR_DATABASE_ERROR", status_code=500)


class ValidationException(RAGSystemException):
    """バリデーション例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="VALIDATION_ERROR", status_code=422)


class AuthenticationException(RAGSystemException):
    """認証例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", status_code=401)


class AuthorizationException(RAGSystemException):
    """認可例外"""

    def __init__(self, message: str):
        super().__init__(message, error_code="AUTHORIZATION_ERROR", status_code=403)
