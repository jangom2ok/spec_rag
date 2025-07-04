# Step07: エラーハンドリングと監視

## 🎯 この章の目標
構造化エラーハンドリング、ログ管理、メトリクス収集、アラート設定、トラブルシューティング手法を理解する

---

## 📋 概要

本番環境でのRAGシステムでは、様々な障害や例外が発生します。適切なエラーハンドリングと包括的な監視により、システムの可用性と信頼性を確保し、問題の早期発見・迅速な対応を実現します。

### 🏗️ 監視・エラーハンドリング構成

```
監視・エラーハンドリング システム
├── エラーハンドリング
│   ├── 例外分類         # ビジネス例外・システム例外
│   ├── 構造化レスポンス # 統一エラーフォーマット  
│   ├── 回復処理         # リトライ・フォールバック
│   └── エラー追跡       # スタックトレース・コンテキスト
├── ログ管理
│   ├── 構造化ログ       # JSON形式・メタデータ付与
│   ├── ログレベル       # DEBUG/INFO/WARN/ERROR/CRITICAL
│   ├── ログローテーション # サイズ・日時ベース回転
│   └── セキュリティログ # 認証・認可・セキュリティイベント
├── メトリクス収集
│   ├── システムメトリクス # CPU/メモリ/ディスク
│   ├── アプリメトリクス   # レスポンス時間・スループット
│   ├── ビジネスメトリクス # 検索精度・ユーザー満足度
│   └── カスタムメトリクス # RAG固有の指標
└── アラート・通知
    ├── 閾値ベースアラート # CPU/メモリ使用率
    ├── 異常検知アラート   # 機械学習ベース
    ├── エラー率アラート   # エラー発生率監視
    └── 外部通知           # Slack/Teams/Email
```

---

## 🚨 エラーハンドリング システム

### 1. 例外分類と階層化

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum

class ErrorCategory(str, Enum):
    """エラーカテゴリ"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication" 
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_ERROR = "system_error"
    RESOURCE_ERROR = "resource_error"

class ErrorSeverity(str, Enum):
    """エラー重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BaseCustomException(Exception, ABC):
    """カスタム例外の基底クラス"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[dict] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
    
    @abstractmethod
    def get_http_status_code(self) -> int:
        """HTTPステータスコード取得"""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "category": self.category.value,
                "severity": self.severity.value,
                "details": self.details
            }
        }

# ビジネスロジック例外
class ValidationError(BaseCustomException):
    def __init__(self, message: str, field: str = None, **kwargs):
        details = {"field": field} if field else {}
        details.update(kwargs.get("details", {}))
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 400

class AuthenticationError(BaseCustomException):
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR", 
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 401

class AuthorizationError(BaseCustomException):
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            category=ErrorCategory.AUTHORIZATION, 
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 403

# システム例外
class EmbeddingServiceError(BaseCustomException):
    def __init__(self, message: str, model_name: str = None, **kwargs):
        details = {"model_name": model_name} if model_name else {}
        details.update(kwargs.get("details", {}))
        
        super().__init__(
            message=message,
            error_code="EMBEDDING_SERVICE_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 503

class VectorDatabaseError(BaseCustomException):
    def __init__(self, message: str, operation: str = None, **kwargs):
        details = {"operation": operation} if operation else {}
        details.update(kwargs.get("details", {}))
        
        super().__init__(
            message=message,
            error_code="VECTOR_DATABASE_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 503

class ResourceExhaustedError(BaseCustomException):
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        details = {"resource_type": resource_type} if resource_type else {}
        details.update(kwargs.get("details", {}))
        
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTED",
            category=ErrorCategory.RESOURCE_ERROR,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            **kwargs
        )
    
    def get_http_status_code(self) -> int:
        return 507
```

### 2. 統一エラーハンドラー

```python
import traceback
import uuid
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

class ErrorHandler:
    """統一エラーハンドラー"""
    
    def __init__(self, logger, metrics_collector):
        self.logger = logger
        self.metrics = metrics_collector
    
    async def handle_custom_exception(
        self,
        request: Request,
        exc: BaseCustomException
    ) -> JSONResponse:
        """カスタム例外ハンドリング"""
        
        # リクエストID生成
        request_id = str(uuid.uuid4())
        
        # エラーログ記録
        await self._log_error(
            exception=exc,
            request=request,
            request_id=request_id
        )
        
        # メトリクス記録
        await self._record_error_metrics(exc, request)
        
        # アラート判定
        await self._check_alert_conditions(exc)
        
        # レスポンス生成
        error_response = {
            **exc.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        return JSONResponse(
            status_code=exc.get_http_status_code(),
            content=error_response
        )
    
    async def handle_validation_exception(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """バリデーション例外ハンドリング"""
        
        request_id = str(uuid.uuid4())
        
        # 詳細なバリデーションエラー情報
        validation_details = []
        for error in exc.errors():
            validation_details.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "category": "validation",
                "severity": "low",
                "details": {
                    "validation_errors": validation_details
                }
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        await self._log_validation_error(exc, request, request_id)
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    async def handle_http_exception(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """HTTP例外ハンドリング"""
        
        request_id = str(uuid.uuid4())
        
        # エラーカテゴリー判定
        category = self._categorize_http_error(exc.status_code)
        severity = self._determine_severity(exc.status_code)
        
        error_response = {
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "category": category,
                "severity": severity,
                "details": {}
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        await self._log_http_error(exc, request, request_id)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def handle_unexpected_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """予期しない例外ハンドリング"""
        
        request_id = str(uuid.uuid4())
        
        # スタックトレース取得
        stack_trace = traceback.format_exc()
        
        # クリティカルログ記録
        self.logger.critical(
            "Unexpected error occurred",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "stack_trace": stack_trace,
                "path": str(request.url.path),
                "method": request.method,
                "user_agent": request.headers.get("User-Agent"),
                "ip_address": request.client.host
            }
        )
        
        # メトリクス記録
        await self.metrics.increment_counter(
            "errors_total",
            labels={
                "category": "system_error",
                "severity": "critical",
                "exception_type": type(exc).__name__
            }
        )
        
        # 緊急アラート
        await self._send_critical_alert(exc, request, request_id)
        
        # セキュアなエラーレスポンス（内部詳細は隠蔽）
        error_response = {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "category": "system_error", 
                "severity": "critical",
                "details": {}
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
```

### 3. リトライ・回復処理

```python
import asyncio
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')

class RetryConfig:
    """リトライ設定"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class CircuitBreakerConfig:
    """サーキットブレーカー設定"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

def retry_with_exponential_backoff(
    retry_config: RetryConfig,
    exceptions: tuple[type] = (Exception,)
):
    """指数バックオフ付きリトライデコレーター"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    # 成功時、前回の失敗をリセット
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded after {attempt + 1} attempts"
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # 最後の試行でない場合のみリトライ
                    if attempt < retry_config.max_attempts - 1:
                        # 指数バックオフ計算
                        delay = min(
                            retry_config.base_delay * (retry_config.exponential_base ** attempt),
                            retry_config.max_delay
                        )
                        
                        # ジッターの追加
                        if retry_config.jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {retry_config.max_attempts} attempts failed for {func.__name__}"
                        )
            
            # 全試行失敗時、最後の例外を再発生
            raise last_exception
        
        return wrapper
    return decorator

class CircuitBreaker:
    """サーキットブレーカー"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """サーキットブレーカー経由での関数呼び出し"""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: Attempting reset")
            else:
                raise EmbeddingServiceError(
                    "Service temporarily unavailable (circuit breaker open)"
                )
        
        try:
            result = await func(*args, **kwargs)
            
            # 成功時の状態更新
            if self.state == "HALF_OPEN":
                self._reset()
                logger.info("Circuit breaker: Reset successful")
            
            return result
            
        except self.config.expected_exception as e:
            self._record_failure()
            
            if self.failure_count >= self.config.failure_threshold:
                self._trip()
                logger.warning("Circuit breaker: Tripped due to failures")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """リセット試行判定"""
        if self.last_failure_time is None:
            return True
        
        return (
            time.time() - self.last_failure_time 
            >= self.config.recovery_timeout
        )
    
    def _record_failure(self):
        """失敗記録"""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def _trip(self):
        """サーキットブレーカーを開く"""
        self.state = "OPEN"
        self.last_failure_time = time.time()
    
    def _reset(self):
        """サーキットブレーカーをリセット"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

# 使用例
@retry_with_exponential_backoff(
    RetryConfig(max_attempts=3, base_delay=1.0),
    exceptions=(VectorDatabaseError, ConnectionError)
)
async def search_vectors_with_retry(query_vector: list[float]):
    """リトライ付きベクター検索"""
    try:
        return await vector_db.search(query_vector)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise VectorDatabaseError(f"Vector search failed: {e}")
```

---

## 📊 ログ管理システム

### 1. 構造化ログ設定

```python
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

class StructuredLogger:
    """構造化ログ管理"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 既存ハンドラーをクリア
        self.logger.handlers.clear()
        
        # JSONフォーマッターの設定
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)
        
        # ファイルハンドラー（本番環境）
        if level.upper() in ["ERROR", "CRITICAL"]:
            error_handler = logging.FileHandler("logs/error.log")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(json_formatter)
            self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **extra):
        """デバッグログ"""
        self._log(logging.DEBUG, message, **extra)
    
    def info(self, message: str, **extra):
        """情報ログ"""
        self._log(logging.INFO, message, **extra)
    
    def warning(self, message: str, **extra):
        """警告ログ"""
        self._log(logging.WARNING, message, **extra)
    
    def error(self, message: str, **extra):
        """エラーログ"""
        self._log(logging.ERROR, message, **extra)
    
    def critical(self, message: str, **extra):
        """クリティカルログ"""
        self._log(logging.CRITICAL, message, **extra)
    
    def _log(self, level: int, message: str, **extra):
        """内部ログ出力"""
        
        # タイムスタンプ追加
        extra['timestamp'] = datetime.utcnow().isoformat()
        
        # ログレベル名追加
        extra['level'] = logging.getLevelName(level)
        
        # サービス情報追加
        extra['service'] = 'rag-system'
        extra['version'] = '1.0.0'
        
        self.logger.log(level, message, extra=extra)

# カスタムログフィルター
class SensitiveDataFilter(logging.Filter):
    """機密データフィルター"""
    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'token["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'api_key["\']?\s*[:=]\s*["\']?([^"\']+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\']+)'
        ]
    
    def filter(self, record):
        """ログレコードの機密データをマスク"""
        
        if hasattr(record, 'msg'):
            for pattern in self.sensitive_patterns:
                record.msg = re.sub(
                    pattern, 
                    r'\1***REDACTED***', 
                    str(record.msg),
                    flags=re.IGNORECASE
                )
        
        return True

# リクエストログミドルウェア
class RequestLoggingMiddleware:
    """リクエストログミドルウェア"""
    
    def __init__(self, app, logger: StructuredLogger):
        self.app = app
        self.logger = logger
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # リクエスト開始ログ
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            self.logger.info(
                "Request started",
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                query_params=str(request.query_params),
                user_agent=request.headers.get("User-Agent"),
                ip_address=request.client.host,
                content_length=request.headers.get("Content-Length")
            )
            
            # レスポンス情報を記録するためのラッパー
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    processing_time = time.time() - start_time
                    
                    # レスポンス完了ログ
                    log_level = "error" if status_code >= 400 else "info"
                    getattr(self.logger, log_level)(
                        "Request completed",
                        request_id=request_id,
                        status_code=status_code,
                        processing_time_ms=round(processing_time * 1000, 2),
                        method=request.method,
                        path=str(request.url.path)
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
```

### 2. セキュリティ監査ログ

```python
class SecurityAuditLogger:
    """セキュリティ監査ログ"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    async def log_authentication_attempt(
        self,
        email: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        failure_reason: str = None
    ):
        """認証試行ログ"""
        
        self.logger.info(
            "Authentication attempt",
            event_type="authentication",
            email=email,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason,
            security_event=True
        )
    
    async def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str
    ):
        """認可失敗ログ"""
        
        self.logger.warning(
            "Authorization failed",
            event_type="authorization_failure",
            user_id=user_id,
            resource=resource,
            action=action,
            ip_address=ip_address,
            security_event=True
        )
    
    async def log_suspicious_activity(
        self,
        user_id: str,
        activity_type: str,
        details: dict,
        risk_score: float,
        ip_address: str
    ):
        """疑わしい活動ログ"""
        
        log_level = "critical" if risk_score > 0.8 else "warning"
        
        getattr(self.logger, log_level)(
            "Suspicious activity detected",
            event_type="suspicious_activity",
            user_id=user_id,
            activity_type=activity_type,
            risk_score=risk_score,
            ip_address=ip_address,
            details=details,
            security_event=True
        )
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        sensitive_data: bool = False
    ):
        """データアクセスログ"""
        
        self.logger.info(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            sensitive_data=sensitive_data,
            audit_event=True
        )
    
    async def log_system_change(
        self,
        user_id: str,
        change_type: str,
        old_value: Any,
        new_value: Any,
        resource: str
    ):
        """システム変更ログ"""
        
        self.logger.info(
            "System configuration changed",
            event_type="system_change",
            user_id=user_id,
            change_type=change_type,
            old_value=str(old_value),
            new_value=str(new_value),
            resource=resource,
            audit_event=True
        )
```

---

## 📈 メトリクス収集システム

### 1. Prometheus メトリクス

```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Dict, List

class MetricsCollector:
    """Prometheusメトリクス収集"""
    
    def __init__(self):
        # カウンター メトリクス
        self.requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.errors_total = Counter(
            'errors_total', 
            'Total errors',
            ['category', 'severity', 'error_code']
        )
        
        self.search_requests_total = Counter(
            'search_requests_total',
            'Total search requests',
            ['search_type', 'user_type']
        )
        
        # ヒストグラム メトリクス
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.search_duration = Histogram(
            'search_duration_seconds',
            'Search operation duration',
            ['search_type'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.embedding_duration = Histogram(
            'embedding_duration_seconds',
            'Embedding generation duration',
            ['model_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # ゲージ メトリクス
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache_type']
        )
        
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['queue_name']
        )
        
        # インフォ メトリクス
        self.system_info = Info(
            'system_info',
            'System information'
        )
        
        # システム情報設定
        self.system_info.info({
            'version': '1.0.0',
            'python_version': sys.version,
            'service': 'rag-system'
        })
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """HTTPリクエストメトリクス記録"""
        
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_error(
        self,
        category: str,
        severity: str,
        error_code: str
    ):
        """エラーメトリクス記録"""
        
        self.errors_total.labels(
            category=category,
            severity=severity,
            error_code=error_code
        ).inc()
    
    def record_search(
        self,
        search_type: str,
        user_type: str,
        duration: float,
        result_count: int
    ):
        """検索メトリクス記録"""
        
        self.search_requests_total.labels(
            search_type=search_type,
            user_type=user_type
        ).inc()
        
        self.search_duration.labels(
            search_type=search_type
        ).observe(duration)
    
    def record_embedding(
        self,
        model_name: str,
        duration: float,
        batch_size: int
    ):
        """埋め込みメトリクス記録"""
        
        self.embedding_duration.labels(
            model_name=model_name
        ).observe(duration)
    
    def update_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """ゲージメトリクス更新"""
        
        gauge = getattr(self, metric_name, None)
        if gauge:
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
```

### 2. カスタムメトリクス

```python
class RAGMetrics:
    """RAG固有のメトリクス"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
        # RAG固有メトリクス
        self.search_precision = Gauge(
            'search_precision',
            'Search precision score',
            ['search_type', 'time_window']
        )
        
        self.search_recall = Gauge(
            'search_recall',
            'Search recall score', 
            ['search_type', 'time_window']
        )
        
        self.user_satisfaction = Gauge(
            'user_satisfaction_score',
            'User satisfaction score',
            ['time_window']
        )
        
        self.document_freshness = Gauge(
            'document_freshness_days',
            'Average document age in days',
            ['source_type']
        )
        
        self.embedding_coverage = Gauge(
            'embedding_coverage_ratio',
            'Ratio of documents with embeddings'
        )
    
    async def calculate_search_quality_metrics(self):
        """検索品質メトリクス計算"""
        
        # 過去24時間の検索ログ分析
        search_logs = await self._get_search_logs(hours=24)
        
        # 検索タイプ別の精度・再現率計算
        for search_type in ['hybrid', 'semantic', 'keyword']:
            type_logs = [log for log in search_logs if log['search_type'] == search_type]
            
            if type_logs:
                precision = await self._calculate_precision(type_logs)
                recall = await self._calculate_recall(type_logs)
                
                self.search_precision.labels(
                    search_type=search_type,
                    time_window='24h'
                ).set(precision)
                
                self.search_recall.labels(
                    search_type=search_type,
                    time_window='24h'
                ).set(recall)
    
    async def calculate_user_satisfaction(self):
        """ユーザー満足度計算"""
        
        # ユーザーフィードバック分析
        feedback_data = await self._get_user_feedback(hours=24)
        
        if feedback_data:
            avg_satisfaction = sum(fb['rating'] for fb in feedback_data) / len(feedback_data)
            
            self.user_satisfaction.labels(
                time_window='24h'
            ).set(avg_satisfaction)
    
    async def calculate_document_freshness(self):
        """ドキュメント鮮度計算"""
        
        # ソースタイプ別の平均文書年数
        source_stats = await self._get_document_age_stats()
        
        for source_type, avg_age_days in source_stats.items():
            self.document_freshness.labels(
                source_type=source_type
            ).set(avg_age_days)
    
    async def _calculate_precision(self, search_logs: List[Dict]) -> float:
        """検索精度計算"""
        
        relevant_results = 0
        total_results = 0
        
        for log in search_logs:
            if 'user_clicks' in log and 'results' in log:
                clicked_positions = log['user_clicks']
                total_results += len(log['results'])
                
                # 上位5件での精度計算
                for pos in clicked_positions:
                    if pos <= 5:  # 上位5件
                        relevant_results += 1
        
        return relevant_results / total_results if total_results > 0 else 0.0
    
    async def _calculate_recall(self, search_logs: List[Dict]) -> float:
        """検索再現率計算"""
        
        # 理想的な結果セットとの比較による再現率計算
        # 実装は業務要件に依存
        
        total_relevant = 0
        retrieved_relevant = 0
        
        for log in search_logs:
            if 'expected_results' in log and 'actual_results' in log:
                expected = set(log['expected_results'])
                actual = set(log['actual_results'])
                
                total_relevant += len(expected)
                retrieved_relevant += len(expected.intersection(actual))
        
        return retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
```

---

## 🚨 アラート・通知システム

### 1. アラート設定

```python
from typing import Callable, List
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """アラートルール定義"""
    
    name: str
    description: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==
    threshold: float
    duration: int   # 秒
    severity: AlertSeverity
    labels: Dict[str, str] = None
    
    def evaluate(self, current_value: float) -> bool:
        """アラート条件評価"""
        
        if self.condition == ">":
            return current_value > self.threshold
        elif self.condition == "<":
            return current_value < self.threshold
        elif self.condition == ">=":
            return current_value >= self.threshold
        elif self.condition == "<=":
            return current_value <= self.threshold
        elif self.condition == "==":
            return abs(current_value - self.threshold) < 0.001
        
        return False

class AlertManager:
    """アラート管理システム"""
    
    def __init__(self, notification_service):
        self.notification_service = notification_service
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, dict] = {}
        self.alert_history: List[dict] = []
    
    def add_rule(self, rule: AlertRule):
        """アラートルール追加"""
        self.alert_rules.append(rule)
    
    async def check_alerts(self, metrics: Dict[str, float]):
        """アラートチェック実行"""
        
        current_time = time.time()
        
        for rule in self.alert_rules:
            metric_value = metrics.get(rule.metric_name)
            
            if metric_value is not None:
                should_alert = rule.evaluate(metric_value)
                alert_key = f"{rule.name}_{rule.metric_name}"
                
                if should_alert:
                    await self._handle_alert_trigger(rule, metric_value, current_time)
                else:
                    await self._handle_alert_resolve(alert_key, current_time)
    
    async def _handle_alert_trigger(
        self,
        rule: AlertRule, 
        current_value: float,
        current_time: float
    ):
        """アラート発火処理"""
        
        alert_key = f"{rule.name}_{rule.metric_name}"
        
        if alert_key in self.active_alerts:
            # 既存アラートの継続時間チェック
            alert_start = self.active_alerts[alert_key]['start_time']
            duration = current_time - alert_start
            
            if duration >= rule.duration:
                # 継続時間を超えた場合、通知送信
                await self._send_alert_notification(rule, current_value, duration)
        else:
            # 新規アラート記録
            self.active_alerts[alert_key] = {
                'rule': rule,
                'start_time': current_time,
                'current_value': current_value,
                'notified': False
            }
    
    async def _handle_alert_resolve(self, alert_key: str, current_time: float):
        """アラート解決処理"""
        
        if alert_key in self.active_alerts:
            alert_info = self.active_alerts[alert_key]
            
            # 解決通知
            if alert_info['notified']:
                await self._send_resolution_notification(alert_info, current_time)
            
            # アラート履歴に追加
            self.alert_history.append({
                'rule_name': alert_info['rule'].name,
                'start_time': alert_info['start_time'],
                'end_time': current_time,
                'duration': current_time - alert_info['start_time'],
                'peak_value': alert_info['current_value']
            })
            
            # アクティブアラートから削除
            del self.active_alerts[alert_key]
    
    async def _send_alert_notification(
        self,
        rule: AlertRule,
        current_value: float,
        duration: float
    ):
        """アラート通知送信"""
        
        message = {
            "title": f"🚨 {rule.severity.value.upper()}: {rule.name}",
            "description": rule.description,
            "current_value": current_value,
            "threshold": rule.threshold,
            "duration": duration,
            "severity": rule.severity.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.notification_service.send_alert(message)
        
        # 通知済みフラグ設定
        alert_key = f"{rule.name}_{rule.metric_name}"
        self.active_alerts[alert_key]['notified'] = True

# システム標準アラートルール
SYSTEM_ALERT_RULES = [
    AlertRule(
        name="High Error Rate",
        description="Error rate exceeds 5%",
        metric_name="error_rate",
        condition=">",
        threshold=0.05,
        duration=300,  # 5分
        severity=AlertSeverity.WARNING
    ),
    AlertRule(
        name="Search Response Time",
        description="Average search response time exceeds 2 seconds",
        metric_name="search_avg_duration",
        condition=">",
        threshold=2.0,
        duration=600,  # 10分
        severity=AlertSeverity.WARNING
    ),
    AlertRule(
        name="Memory Usage Critical",
        description="Memory usage exceeds 90%",
        metric_name="memory_usage_ratio",
        condition=">",
        threshold=0.90,
        duration=60,  # 1分
        severity=AlertSeverity.CRITICAL
    ),
    AlertRule(
        name="Embedding Service Down",
        description="Embedding service unavailable",
        metric_name="embedding_service_availability",
        condition="<",
        threshold=0.5,
        duration=120,  # 2分
        severity=AlertSeverity.CRITICAL
    )
]
```

### 2. 通知サービス

```python
import aiohttp
from typing import Dict, List

class NotificationService:
    """通知サービス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_urls = config.get('webhook_urls', {})
        self.email_config = config.get('email', {})
    
    async def send_alert(self, alert_data: Dict[str, Any]):
        """アラート通知送信"""
        
        # Slack通知
        if 'slack' in self.webhook_urls:
            await self._send_slack_notification(alert_data)
        
        # Teams通知
        if 'teams' in self.webhook_urls:
            await self._send_teams_notification(alert_data)
        
        # Email通知
        if alert_data['severity'] == 'critical':
            await self._send_email_notification(alert_data)
    
    async def _send_slack_notification(self, alert_data: Dict[str, Any]):
        """Slack通知送信"""
        
        color_map = {
            'info': '#36a64f',
            'warning': '#ff9500', 
            'critical': '#ff0000'
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert_data['severity'], '#36a64f'),
                "title": alert_data['title'],
                "text": alert_data['description'],
                "fields": [
                    {
                        "title": "Current Value",
                        "value": str(alert_data['current_value']),
                        "short": True
                    },
                    {
                        "title": "Threshold", 
                        "value": str(alert_data['threshold']),
                        "short": True
                    },
                    {
                        "title": "Duration",
                        "value": f"{alert_data['duration']:.0f} seconds",
                        "short": True
                    }
                ],
                "footer": "RAG System Monitoring",
                "ts": int(time.time())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_urls['slack'],
                json=payload
            ) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
    
    async def _send_email_notification(self, alert_data: Dict[str, Any]):
        """Email通知送信"""
        
        # 実装は使用するEmailサービスに依存
        # AWS SES、SendGrid等を使用
        
        subject = f"RAG System Alert: {alert_data['title']}"
        body = f"""
        Alert: {alert_data['title']}
        Description: {alert_data['description']}
        Current Value: {alert_data['current_value']}
        Threshold: {alert_data['threshold']}
        Duration: {alert_data['duration']} seconds
        Timestamp: {alert_data['timestamp']}
        """
        
        # Email送信処理
        logger.info(f"Critical alert email sent: {subject}")
```

---

## ❗ よくある落とし穴と対策

### 1. ログの機密情報漏洩

```python
# ❌ 問題: パスワードやトークンをログ出力
logger.info(f"User login: {request_data}")  # パスワード含む

# ✅ 対策: 機密データフィルタリング
def sanitize_log_data(data: dict) -> dict:
    """ログデータの機密情報除去"""
    sensitive_keys = ['password', 'token', 'api_key', 'secret']
    
    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '***REDACTED***'
    
    return sanitized

logger.info(f"User login: {sanitize_log_data(request_data)}")
```

### 2. 無制限ログ蓄積

```python
# ❌ 問題: ログローテーション設定なし
# ログファイルが無制限に増大

# ✅ 対策: ログローテーション設定
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10           # 10ファイル保持
)
```

### 3. アラート疲れ

```python
# ❌ 問題: 過敏なアラート設定
AlertRule(
    threshold=0.01,  # 1%エラー率で即アラート
    duration=10      # 10秒継続で通知
)

# ✅ 対策: 適切な閾値・継続時間設定
AlertRule(
    threshold=0.05,  # 5%エラー率
    duration=300,    # 5分継続
    cooldown=1800    # 30分のクールダウン
)
```

---

## 🎯 理解確認のための設問

### エラーハンドリング理解
1. カスタム例外でエラーカテゴリとエラー重要度を分ける理由を説明してください
2. リトライ処理で指数バックオフとジッターを使用する目的を説明してください
3. サーキットブレーカーパターンが有効な場面と状態遷移を説明してください

### ログ管理理解
1. 構造化ログ（JSON形式）の利点と、必須で含めるべき情報を5つ挙げてください
2. セキュリティ監査ログで記録すべきイベントを6つ挙げてください
3. ログレベル（DEBUG/INFO/WARN/ERROR/CRITICAL）の使い分け基準を説明してください

### メトリクス理解
1. Counter、Gauge、Histogramメトリクスの違いと適用場面を説明してください
2. RAGシステム特有のメトリクス（検索精度・再現率）の計算方法を説明してください
3. アラートルールで継続時間（duration）を設定する理由と適切な値の決め方を説明してください

### 監視・運用理解
1. 効果的なアラート設計で避けるべき3つの問題とその対策を説明してください
2. 障害対応時に確認すべきログ・メトリクスの優先順位を説明してください
3. システムの可観測性（Observability）を向上させるための3つの要素を説明してください

---

## 📚 次のステップ

エラーハンドリングと監視を理解できたら、最終段階に進んでください：

- **Step08**: デプロイメントと運用 - Docker・Kubernetes・CI/CD・本番運用

適切なエラーハンドリングと監視は、システムの信頼性と運用効率を大幅に向上させます。次のステップでは、これらの実装をコンテナ環境にデプロイし、本番運用するための手法を学習します。