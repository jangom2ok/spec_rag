"""メトリクス収集サービス

TDD実装：監視・メトリクス収集機能
- 検索メトリクス: レスポンス時間、検索頻度、成功率
- ユーザーメトリクス: アクティブユーザー、セッション時間
- システムメトリクス: CPU、メモリ、ディスク使用率
- パフォーマンスメトリクス: スループット、エラー率
- ビジネスメトリクス: クエリ品質、満足度指標
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """メトリクスタイプ"""

    SEARCH_METRICS = "search_metrics"
    USER_METRICS = "user_metrics"
    SYSTEM_METRICS = "system_metrics"
    PERFORMANCE_METRICS = "performance_metrics"
    BUSINESS_METRICS = "business_metrics"


class AggregationType(str, Enum):
    """集約タイプ"""

    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    P99 = "p99"


class TimeWindow(str, Enum):
    """時間ウィンドウ"""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class MetricsConfig:
    """メトリクス設定"""

    metric_types: list[MetricType]
    collection_interval: int = 60  # 秒
    retention_days: int = 30

    # リアルタイム設定
    enable_real_time: bool = False
    real_time_buffer_size: int = 1000

    # 集約設定
    enable_aggregation: bool = False
    aggregation_interval: int = 300  # 秒
    aggregation_types: list[AggregationType] = field(
        default_factory=lambda: [AggregationType.AVG, AggregationType.COUNT]
    )

    # アラート設定
    enable_alerting: bool = False
    alert_thresholds: dict[str, float] = field(default_factory=dict)

    # ストレージ設定
    storage_backend: str = "memory"  # memory, redis, database
    max_memory_usage: int = 1024 * 1024 * 100  # 100MB

    # パフォーマンス設定
    batch_size: int = 100
    flush_interval: int = 60
    max_concurrent_operations: int = 10

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.collection_interval <= 0:
            raise ValueError("collection_interval must be greater than 0")
        if self.retention_days <= 0:
            raise ValueError("retention_days must be greater than 0")
        if not self.metric_types:
            raise ValueError("metric_types cannot be empty")


@dataclass
class MetricFilter:
    """メトリクスフィルター"""

    field: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, not_in


@dataclass
class MetricsRequest:
    """メトリクス取得リクエスト"""

    metric_types: list[MetricType]
    start_time: datetime
    end_time: datetime
    time_window: TimeWindow = TimeWindow.HOUR
    aggregation_type: AggregationType = AggregationType.AVG
    filters: list[MetricFilter] = field(default_factory=list)
    max_results: int = 1000
    include_raw: bool = False


@dataclass
class MetricData:
    """基底メトリクスデータ"""

    timestamp: datetime
    metric_type: MetricType
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type,
            "data": self.data,
        }


@dataclass
class SearchMetrics:
    """検索メトリクス"""

    timestamp: datetime
    user_id: str
    session_id: str
    query: str
    response_time: float  # ミリ秒
    result_count: int
    clicked_position: int | None
    is_successful: bool
    search_mode: str
    error_message: str | None = None

    @property
    def click_through_rate(self) -> float:
        """クリックスルー率"""
        return 1.0 if self.clicked_position is not None else 0.0


@dataclass
class UserMetrics:
    """ユーザーメトリクス"""

    timestamp: datetime
    user_id: str
    session_id: str
    action: str
    session_duration: float | None = None
    page_views: int = 0
    search_count: int = 0
    click_count: int = 0


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: datetime
    cpu_usage: float  # 0.0 - 1.0
    memory_usage: float  # 0.0 - 1.0
    disk_usage: float  # 0.0 - 1.0
    network_io_bytes: int
    active_connections: int = 0


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    timestamp: datetime
    endpoint: str
    response_time: float  # ミリ秒
    request_count: int
    error_count: int
    throughput: float  # リクエスト/秒

    @property
    def error_rate(self) -> float:
        """エラー率"""
        return self.error_count / max(self.request_count, 1)


@dataclass
class BusinessMetrics:
    """ビジネスメトリクス"""

    timestamp: datetime
    query_quality_score: float  # 0.0 - 1.0
    user_satisfaction_score: float  # 1.0 - 5.0
    conversion_rate: float  # 0.0 - 1.0
    engagement_rate: float  # 0.0 - 1.0
    retention_rate: float  # 0.0 - 1.0


@dataclass
class MetricsResponse:
    """メトリクス応答"""

    success: bool
    data: list[dict[str, Any]]
    total_count: int
    time_window: TimeWindow
    aggregation_type: AggregationType
    processing_time: float
    error_message: str | None = None

    def get_summary(self) -> dict[str, Any]:
        """応答サマリーを取得"""
        return {
            "success": self.success,
            "total_count": self.total_count,
            "time_window": self.time_window,
            "aggregation_type": self.aggregation_type,
            "processing_time": self.processing_time,
        }


class MetricsCollectionService:
    """メトリクス収集サービスメインクラス"""

    def __init__(self, config: MetricsConfig):
        self.config = config

        # 収集状態管理
        self._is_collecting = False
        self._collection_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # メトリクスバッファ
        self._search_metrics_buffer: deque[SearchMetrics] = deque(
            maxlen=config.real_time_buffer_size
        )
        self._user_metrics_buffer: deque[UserMetrics] = deque(
            maxlen=config.real_time_buffer_size
        )
        self._system_metrics_buffer: deque[SystemMetrics] = deque(
            maxlen=config.real_time_buffer_size
        )
        self._performance_metrics_buffer: deque[PerformanceMetrics] = deque(
            maxlen=config.real_time_buffer_size
        )
        self._business_metrics_buffer: deque[BusinessMetrics] = deque(
            maxlen=config.real_time_buffer_size
        )

        # コレクター管理
        self._metric_collectors: dict[MetricType, Callable] = {}

        # セッション管理
        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._user_activities: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # リアルタイムコールバック
        self._real_time_callback: Callable[[dict[str, Any]], None] | None = None

        # 集約データキャッシュ
        self._aggregated_cache: dict[str, Any] = {}

        # アラート状態
        self._alert_states: dict[str, dict[str, Any]] = {}

    async def start_collection(self) -> None:
        """メトリクス収集開始"""
        if self._is_collecting:
            logger.warning("Metrics collection is already running")
            return

        logger.info("Starting metrics collection")
        self._is_collecting = True

        # メトリクスコレクターの初期化
        await self._initialize_collectors()

        # バックグラウンド収集スレッドの開始
        if self.config.enable_real_time:
            self._collection_thread = threading.Thread(
                target=self._background_collection_loop, daemon=True
            )
            self._collection_thread.start()

    async def stop_collection(self) -> None:
        """メトリクス収集停止"""
        if not self._is_collecting:
            return

        logger.info("Stopping metrics collection")
        self._is_collecting = False
        self._stop_event.set()

        # スレッド終了待機
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)

    async def _initialize_collectors(self) -> None:
        """コレクターの初期化"""
        for metric_type in self.config.metric_types:
            if metric_type == MetricType.SEARCH_METRICS:
                self._metric_collectors[metric_type] = self._collect_search_metrics
            elif metric_type == MetricType.USER_METRICS:
                self._metric_collectors[metric_type] = self._collect_user_metrics
            elif metric_type == MetricType.SYSTEM_METRICS:
                self._metric_collectors[metric_type] = self._collect_system_metrics
            elif metric_type == MetricType.PERFORMANCE_METRICS:
                self._metric_collectors[metric_type] = self._collect_performance_metrics
            elif metric_type == MetricType.BUSINESS_METRICS:
                self._metric_collectors[metric_type] = self._collect_business_metrics

    def _background_collection_loop(self) -> None:
        """バックグラウンド収集ループ"""
        while not self._stop_event.wait(self.config.collection_interval):
            try:
                # システムメトリクスの定期収集
                if MetricType.SYSTEM_METRICS in self.config.metric_types:
                    asyncio.run_coroutine_threadsafe(
                        self._collect_and_store_system_metrics(),
                        asyncio.get_event_loop(),
                    )

                # 集約処理の実行
                if self.config.enable_aggregation:
                    asyncio.run_coroutine_threadsafe(
                        self._perform_aggregation(), asyncio.get_event_loop()
                    )

                # バッファフラッシュ
                asyncio.run_coroutine_threadsafe(
                    self._flush_buffers(), asyncio.get_event_loop()
                )

            except Exception as e:
                logger.error(f"Background collection error: {e}")

    async def record_search_metric(self, metric: SearchMetrics) -> None:
        """検索メトリクス記録"""
        try:
            self._search_metrics_buffer.append(metric)

            # リアルタイムコールバック
            if self._real_time_callback and self.config.enable_real_time:
                await self._trigger_real_time_callback("search_metric", asdict(metric))

            # アラートチェック
            if self.config.enable_alerting:
                await self._check_search_metric_alerts(metric)

        except Exception as e:
            logger.error(f"Failed to record search metric: {e}")

    async def record_user_metric(self, metric: UserMetrics) -> None:
        """ユーザーメトリクス記録"""
        try:
            self._user_metrics_buffer.append(metric)

            if self._real_time_callback and self.config.enable_real_time:
                await self._trigger_real_time_callback("user_metric", asdict(metric))

        except Exception as e:
            logger.error(f"Failed to record user metric: {e}")

    async def record_system_metric(self, metric: SystemMetrics) -> None:
        """システムメトリクス記録"""
        try:
            self._system_metrics_buffer.append(metric)

            if self._real_time_callback and self.config.enable_real_time:
                await self._trigger_real_time_callback("system_metric", asdict(metric))

            # システムアラートチェック
            if self.config.enable_alerting:
                alerts = await self._check_metric_thresholds(metric)
                for alert in alerts:
                    await self._trigger_alert(alert)

        except Exception as e:
            logger.error(f"Failed to record system metric: {e}")

    async def record_performance_metric(self, metric: PerformanceMetrics) -> None:
        """パフォーマンスメトリクス記録"""
        try:
            self._performance_metrics_buffer.append(metric)

            if self._real_time_callback and self.config.enable_real_time:
                await self._trigger_real_time_callback(
                    "performance_metric", asdict(metric)
                )

        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")

    async def record_business_metric(self, metric: BusinessMetrics) -> None:
        """ビジネスメトリクス記録"""
        try:
            self._business_metrics_buffer.append(metric)

            if self._real_time_callback and self.config.enable_real_time:
                await self._trigger_real_time_callback(
                    "business_metric", asdict(metric)
                )

        except Exception as e:
            logger.error(f"Failed to record business metric: {e}")

    async def _collect_search_metrics(self) -> list[SearchMetrics]:
        """検索メトリクス収集"""
        # 実装では実際の検索ログから収集
        return list(self._search_metrics_buffer)

    async def _collect_user_metrics(self) -> list[UserMetrics]:
        """ユーザーメトリクス収集"""
        # セッション情報から現在のユーザーメトリクスを生成
        metrics = []
        current_time = datetime.now()

        for session_id, session_info in self._active_sessions.items():
            user_id = session_info["user_id"]
            start_time = session_info["start_time"]
            session_duration = (current_time - start_time).total_seconds()

            activities = self._user_activities.get(user_id, [])
            search_count = len([a for a in activities if a["action"] == "search"])
            click_count = len([a for a in activities if a["action"] == "click"])

            metric = UserMetrics(
                timestamp=current_time,
                user_id=user_id,
                session_id=session_id,
                action="session_update",
                session_duration=session_duration,
                page_views=session_info.get("page_views", 0),
                search_count=search_count,
                click_count=click_count,
            )
            metrics.append(metric)

        return metrics

    async def _collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net_io = psutil.net_io_counters()

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent / 100.0,
                disk_usage=disk.percent / 100.0,
                network_io_bytes=net_io.bytes_sent + net_io.bytes_recv,
                active_connections=len(psutil.net_connections()),
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # フォールバック値
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io_bytes=0,
                active_connections=0,
            )

    async def _collect_performance_metrics(self) -> list[PerformanceMetrics]:
        """パフォーマンスメトリクス収集"""
        return list(self._performance_metrics_buffer)

    async def _collect_business_metrics(self) -> list[BusinessMetrics]:
        """ビジネスメトリクス収集"""
        return list(self._business_metrics_buffer)

    async def _collect_and_store_system_metrics(self) -> None:
        """システムメトリクス収集・保存"""
        metric = await self._collect_system_metrics()
        await self.record_system_metric(metric)

    async def query_metrics(self, request: MetricsRequest) -> MetricsResponse:
        """メトリクス問い合わせ"""
        start_time = time.time()

        try:
            all_data = []

            for metric_type in request.metric_types:
                data = await self._query_metric_type(metric_type, request)
                all_data.extend(data)

            # フィルタリング適用
            if request.filters:
                all_data = self._apply_filters(all_data, request.filters)

            # 時間範囲フィルタリング
            all_data = self._filter_by_time_range(
                all_data, request.start_time, request.end_time
            )

            # 集約処理
            all_data = await self._aggregate_metrics(
                all_data, request.aggregation_type, request.time_window
            )

            # 結果制限
            if len(all_data) > request.max_results:
                all_data = all_data[: request.max_results]

            processing_time = time.time() - start_time

            return MetricsResponse(
                success=True,
                data=all_data,
                total_count=len(all_data),
                time_window=request.time_window,
                aggregation_type=request.aggregation_type,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to query metrics: {e}")
            processing_time = time.time() - start_time

            return MetricsResponse(
                success=False,
                data=[],
                total_count=0,
                time_window=request.time_window,
                aggregation_type=request.aggregation_type,
                processing_time=processing_time,
                error_message=str(e),
            )

    async def _query_metric_type(
        self, metric_type: MetricType, request: MetricsRequest
    ) -> list[dict[str, Any]]:
        """特定メトリクスタイプの問い合わせ"""
        if metric_type == MetricType.SEARCH_METRICS:
            return [self._search_metric_to_dict(m) for m in self._search_metrics_buffer]
        elif metric_type == MetricType.USER_METRICS:
            return [self._user_metric_to_dict(m) for m in self._user_metrics_buffer]
        elif metric_type == MetricType.SYSTEM_METRICS:
            return [self._system_metric_to_dict(m) for m in self._system_metrics_buffer]
        elif metric_type == MetricType.PERFORMANCE_METRICS:
            return [
                self._performance_metric_to_dict(m)
                for m in self._performance_metrics_buffer
            ]
        elif metric_type == MetricType.BUSINESS_METRICS:
            return [
                self._business_metric_to_dict(m) for m in self._business_metrics_buffer
            ]
        else:
            return []  # type: ignore[unreachable]

    def _search_metric_to_dict(self, metric: SearchMetrics) -> dict[str, Any]:
        """検索メトリクスを辞書に変換"""
        return asdict(metric)

    def _user_metric_to_dict(self, metric: UserMetrics) -> dict[str, Any]:
        """ユーザーメトリクスを辞書に変換"""
        return asdict(metric)

    def _system_metric_to_dict(self, metric: SystemMetrics) -> dict[str, Any]:
        """システムメトリクスを辞書に変換"""
        return asdict(metric)

    def _performance_metric_to_dict(self, metric: PerformanceMetrics) -> dict[str, Any]:
        """パフォーマンスメトリクスを辞書に変換"""
        data = asdict(metric)
        data["error_rate"] = metric.error_rate
        return data

    def _business_metric_to_dict(self, metric: BusinessMetrics) -> dict[str, Any]:
        """ビジネスメトリクスを辞書に変換"""
        return asdict(metric)

    def _apply_filters(
        self, data: list[dict[str, Any]], filters: list[MetricFilter]
    ) -> list[dict[str, Any]]:
        """フィルターを適用"""
        filtered_data = data

        for filter_obj in filters:
            filtered_data = self._apply_single_filter(filtered_data, filter_obj)

        return filtered_data

    def _apply_single_filter(
        self, data: list[dict[str, Any]], filter_obj: MetricFilter
    ) -> list[dict[str, Any]]:
        """単一フィルターを適用"""
        filtered = []

        for item in data:
            value = item.get(filter_obj.field)

            if filter_obj.operator == "eq" and value == filter_obj.value:
                filtered.append(item)
            elif filter_obj.operator == "ne" and value != filter_obj.value:
                filtered.append(item)
            elif (
                filter_obj.operator == "gt"
                and value is not None
                and value > filter_obj.value
            ):
                filtered.append(item)
            elif (
                filter_obj.operator == "lt"
                and value is not None
                and value < filter_obj.value
            ):
                filtered.append(item)
            elif (
                filter_obj.operator == "gte"
                and value is not None
                and value >= filter_obj.value
            ):
                filtered.append(item)
            elif (
                filter_obj.operator == "lte"
                and value is not None
                and value <= filter_obj.value
            ):
                filtered.append(item)
            elif filter_obj.operator == "in" and value in filter_obj.value:
                filtered.append(item)
            elif filter_obj.operator == "not_in" and value not in filter_obj.value:
                filtered.append(item)

        return filtered

    def _filter_by_time_range(
        self, data: list[dict[str, Any]], start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """時間範囲でフィルタリング"""
        filtered = []

        for item in data:
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    # ISO形式とその他の形式に対応
                    if timestamp.endswith("Z"):
                        timestamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    else:
                        timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue
            elif isinstance(timestamp, datetime):
                pass
            else:
                continue

            # タイムゾーン情報が異なる場合の処理
            if timestamp.tzinfo is None and start_time.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=start_time.tzinfo)
            elif timestamp.tzinfo is not None and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timestamp.tzinfo)
                end_time = end_time.replace(tzinfo=timestamp.tzinfo)

            if start_time <= timestamp <= end_time:
                filtered.append(item)

        return filtered

    async def _aggregate_metrics(
        self,
        data: list[dict[str, Any]],
        aggregation_type: AggregationType,
        time_window: TimeWindow,
    ) -> list[dict[str, Any]]:
        """メトリクス集約"""
        if not data:
            return []

        # 時間ウィンドウごとにグループ化
        time_groups = self._group_by_time_window(data, time_window)

        aggregated = []
        for time_key, group_data in time_groups.items():
            aggregated_item = await self._aggregate_group(group_data, aggregation_type)
            aggregated_item["timestamp"] = time_key
            aggregated.append(aggregated_item)

        return aggregated

    def _group_by_time_window(
        self, data: list[dict[str, Any]], time_window: TimeWindow
    ) -> dict[str, list[dict[str, Any]]]:
        """時間ウィンドウでグループ化"""
        groups = defaultdict(list)

        for item in data:
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
            elif isinstance(timestamp, datetime):
                pass
            else:
                continue

            time_key = self._get_time_window_key(timestamp, time_window)
            groups[time_key].append(item)

        return groups

    def _get_time_window_key(self, timestamp: datetime, time_window: TimeWindow) -> str:
        """時間ウィンドウのキーを取得"""
        if time_window == TimeWindow.MINUTE:
            return timestamp.strftime("%Y-%m-%d %H:%M")
        elif time_window == TimeWindow.HOUR:
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif time_window == TimeWindow.DAY:
            return timestamp.strftime("%Y-%m-%d")
        elif time_window == TimeWindow.WEEK:
            # 週の始まりを月曜日とする
            monday = timestamp - timedelta(days=timestamp.weekday())
            return monday.strftime("%Y-%m-%d")
        elif time_window == TimeWindow.MONTH:
            return timestamp.strftime("%Y-%m")
        else:
            return timestamp.isoformat()  # type: ignore[unreachable]

    async def _aggregate_group(
        self, group_data: list[dict[str, Any]], aggregation_type: AggregationType
    ) -> dict[str, Any]:
        """グループデータの集約"""
        if not group_data:
            return {}

        aggregated = {"count": len(group_data)}

        # 数値フィールドを特定
        numeric_fields = self._get_numeric_fields(group_data)

        for field_name in numeric_fields:
            values = [
                item[field_name]
                for item in group_data
                if field_name in item and item[field_name] is not None
            ]

            if not values:
                continue

            if aggregation_type == AggregationType.SUM:
                aggregated[f"sum_{field_name}"] = sum(values)
            elif aggregation_type == AggregationType.AVG:
                aggregated[f"avg_{field_name}"] = mean(values)
            elif aggregation_type == AggregationType.MIN:
                aggregated[f"min_{field_name}"] = min(values)
            elif aggregation_type == AggregationType.MAX:
                aggregated[f"max_{field_name}"] = max(values)
            elif aggregation_type == AggregationType.P95:
                sorted_values = sorted(values)
                p95_index = int(len(sorted_values) * 0.95)
                aggregated[f"p95_{field_name}"] = (
                    sorted_values[p95_index]
                    if p95_index < len(sorted_values)
                    else sorted_values[-1]
                )
            elif aggregation_type == AggregationType.P99:
                sorted_values = sorted(values)
                p99_index = int(len(sorted_values) * 0.99)
                aggregated[f"p99_{field_name}"] = (
                    sorted_values[p99_index]
                    if p99_index < len(sorted_values)
                    else sorted_values[-1]
                )

        return aggregated

    def _get_numeric_fields(self, group_data: list[dict[str, Any]]) -> list[str]:
        """数値フィールドを特定"""
        if not group_data:
            return []

        numeric_fields = []
        sample_item = group_data[0]

        for key, value in sample_item.items():
            if isinstance(value, int | float) and key != "timestamp":
                numeric_fields.append(key)

        return numeric_fields

    async def _aggregate_search_metrics(
        self, metrics: list[SearchMetrics], aggregation_type: AggregationType
    ) -> dict[str, Any]:
        """検索メトリクスの集約"""
        if not metrics:
            return {}

        response_times = [m.response_time for m in metrics]
        successful_searches = [m for m in metrics if m.is_successful]

        aggregated = {
            "count": len(metrics),
            "success_rate": len(successful_searches) / len(metrics),
        }

        if aggregation_type == AggregationType.AVG:
            aggregated["avg_response_time"] = mean(response_times)
        elif aggregation_type == AggregationType.SUM:
            aggregated["sum_response_time"] = sum(response_times)
        elif aggregation_type == AggregationType.MIN:
            aggregated["min_response_time"] = min(response_times)
        elif aggregation_type == AggregationType.MAX:
            aggregated["max_response_time"] = max(response_times)

        return aggregated

    async def _aggregate_by_time_window(
        self,
        metrics: list[SearchMetrics],
        time_window: TimeWindow,
        aggregation_type: AggregationType,
    ) -> list[dict[str, Any]]:
        """時間ウィンドウ別集約"""
        # メトリクスを辞書形式に変換
        metrics_dict = [asdict(m) for m in metrics]

        # 時間ウィンドウでグループ化
        time_groups = self._group_by_time_window(metrics_dict, time_window)

        aggregated = []
        for time_key, group_data in time_groups.items():
            # SearchMetrics形式に戻す
            group_metrics = [
                SearchMetrics(
                    timestamp=(
                        datetime.fromisoformat(item["timestamp"])
                        if isinstance(item["timestamp"], str)
                        else item["timestamp"]
                    ),
                    user_id=item["user_id"],
                    session_id=item["session_id"],
                    query=item["query"],
                    response_time=item["response_time"],
                    result_count=item["result_count"],
                    clicked_position=item["clicked_position"],
                    is_successful=item["is_successful"],
                    search_mode=item["search_mode"],
                    error_message=item.get("error_message"),
                )
                for item in group_data
            ]

            aggregated_item = await self._aggregate_search_metrics(
                group_metrics, aggregation_type
            )
            aggregated_item["timestamp"] = time_key
            aggregated.append(aggregated_item)

        return aggregated

    async def _calculate_average_throughput(
        self, metrics: list[PerformanceMetrics], time_window: timedelta
    ) -> float:
        """平均スループット計算"""
        if not metrics:
            return 0.0

        # 時間ウィンドウ内のメトリクスをフィルタリング
        now = datetime.now()
        cutoff_time = now - time_window

        relevant_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not relevant_metrics:
            return 0.0

        return mean([m.throughput for m in relevant_metrics])

    async def _calculate_average_satisfaction(
        self, metrics: list[BusinessMetrics]
    ) -> float:
        """平均満足度計算"""
        if not metrics:
            return 0.0

        satisfaction_scores = [m.user_satisfaction_score for m in metrics]
        return mean(satisfaction_scores)

    async def start_user_session(self, user_id: str, session_id: str) -> None:
        """ユーザーセッション開始"""
        self._active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "page_views": 0,
        }

    async def end_user_session(self, session_id: str) -> None:
        """ユーザーセッション終了"""
        if session_id in self._active_sessions:
            session_info = self._active_sessions[session_id]
            end_time = datetime.now()
            duration = (end_time - session_info["start_time"]).total_seconds()

            # セッション終了メトリクス記録
            metric = UserMetrics(
                timestamp=end_time,
                user_id=session_info["user_id"],
                session_id=session_id,
                action="session_end",
                session_duration=duration,
                page_views=session_info["page_views"],
            )
            await self.record_user_metric(metric)

            del self._active_sessions[session_id]

    async def record_user_activity(
        self, user_id: str, action: str, data: dict[str, Any]
    ) -> None:
        """ユーザー活動記録"""
        activity = {
            "timestamp": datetime.now(),
            "action": action,
            "data": data,
        }
        self._user_activities[user_id].append(activity)

        # 古い活動履歴の削除（メモリ管理）
        max_activities = 1000
        if len(self._user_activities[user_id]) > max_activities:
            self._user_activities[user_id] = self._user_activities[user_id][
                -max_activities:
            ]

    async def _check_metric_thresholds(
        self, metric: SystemMetrics
    ) -> list[dict[str, Any]]:
        """メトリクス閾値チェック"""
        alerts = []

        # CPU使用率チェック
        if "cpu_usage" in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds["cpu_usage"]
            if metric.cpu_usage > threshold:
                alerts.append(
                    {
                        "metric": "cpu_usage",
                        "value": metric.cpu_usage,
                        "threshold": threshold,
                        "severity": (
                            "warning"
                            if metric.cpu_usage < threshold * 1.2
                            else "critical"
                        ),
                        "timestamp": metric.timestamp,
                    }
                )

        # メモリ使用率チェック
        if "memory_usage" in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds["memory_usage"]
            if metric.memory_usage > threshold:
                alerts.append(
                    {
                        "metric": "memory_usage",
                        "value": metric.memory_usage,
                        "threshold": threshold,
                        "severity": (
                            "warning"
                            if metric.memory_usage < threshold * 1.2
                            else "critical"
                        ),
                        "timestamp": metric.timestamp,
                    }
                )

        return alerts

    async def _check_search_metric_alerts(self, metric: SearchMetrics) -> None:
        """検索メトリクスアラートチェック"""
        # レスポンス時間チェック
        if "response_time_p95" in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds["response_time_p95"]
            if metric.response_time > threshold:
                alert = {
                    "metric": "response_time",
                    "value": metric.response_time,
                    "threshold": threshold,
                    "severity": "warning",
                    "timestamp": metric.timestamp,
                    "query": metric.query,
                }
                await self._trigger_alert(alert)

    async def _trigger_alert(self, alert: dict[str, Any]) -> None:
        """アラート発動"""
        alert_key = f"{alert['metric']}_{alert['severity']}"

        # アラート重複防止
        if alert_key in self._alert_states:
            last_alert = self._alert_states[alert_key]
            time_diff = (alert["timestamp"] - last_alert["timestamp"]).total_seconds()
            if time_diff < 300:  # 5分以内の重複アラートはスキップ
                return

        self._alert_states[alert_key] = alert
        logger.warning(f"ALERT: {alert}")

    async def _trigger_real_time_callback(
        self, event_type: str, data: dict[str, Any]
    ) -> None:
        """リアルタイムコールバック発動"""
        if self._real_time_callback:
            try:
                callback_data = {
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                }
                if asyncio.iscoroutinefunction(self._real_time_callback):
                    await self._real_time_callback(callback_data)
                else:
                    self._real_time_callback(callback_data)
            except Exception as e:
                logger.error(f"Real-time callback error: {e}")

    def set_real_time_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """リアルタイムコールバック設定"""
        self._real_time_callback = callback

    async def _perform_aggregation(self) -> None:
        """定期集約処理"""
        try:
            # 各メトリクスタイプの集約実行
            for metric_type in self.config.metric_types:
                await self._aggregate_metric_type(metric_type)

        except Exception as e:
            logger.error(f"Aggregation error: {e}")

    async def _aggregate_metric_type(self, metric_type: MetricType) -> None:
        """特定メトリクスタイプの集約"""
        cache_key = f"aggregated_{metric_type}_{datetime.now().strftime('%Y%m%d%H%M')}"

        if metric_type == MetricType.SEARCH_METRICS:
            search_metrics = list(self._search_metrics_buffer)
            aggregated = await self._aggregate_search_metrics(
                search_metrics, AggregationType.AVG
            )
        elif metric_type == MetricType.SYSTEM_METRICS:
            system_metrics = list(self._system_metrics_buffer)
            aggregated = {"count": len(system_metrics)}
            if system_metrics:
                aggregated["avg_cpu_usage"] = mean(
                    [m.cpu_usage for m in system_metrics]
                )
                aggregated["avg_memory_usage"] = mean(
                    [m.memory_usage for m in system_metrics]
                )
        else:
            return

        self._aggregated_cache[cache_key] = aggregated

    async def _flush_buffers(self) -> None:
        """バッファフラッシュ"""
        # バッファサイズ制限
        max_buffer_size = self.config.real_time_buffer_size

        # 各バッファのサイズをチェックしてフラッシュ
        if len(self._search_metrics_buffer) > max_buffer_size:
            for _ in range(len(self._search_metrics_buffer) - max_buffer_size):
                self._search_metrics_buffer.popleft()

        if len(self._user_metrics_buffer) > max_buffer_size:
            for _ in range(len(self._user_metrics_buffer) - max_buffer_size):
                self._user_metrics_buffer.popleft()

        if len(self._system_metrics_buffer) > max_buffer_size:
            for _ in range(len(self._system_metrics_buffer) - max_buffer_size):
                self._system_metrics_buffer.popleft()

        if len(self._performance_metrics_buffer) > max_buffer_size:
            for _ in range(len(self._performance_metrics_buffer) - max_buffer_size):
                self._performance_metrics_buffer.popleft()

        if len(self._business_metrics_buffer) > max_buffer_size:
            for _ in range(len(self._business_metrics_buffer) - max_buffer_size):
                self._business_metrics_buffer.popleft()

    async def export_metrics(
        self, format: str = "json", filters: list[MetricFilter] | None = None
    ) -> str:
        """メトリクスエクスポート"""
        export_data = {}

        # 各メトリクスタイプのデータ収集
        export_data["search_metrics"] = [asdict(m) for m in self._search_metrics_buffer]
        export_data["user_metrics"] = [asdict(m) for m in self._user_metrics_buffer]
        export_data["system_metrics"] = [asdict(m) for m in self._system_metrics_buffer]
        export_data["performance_metrics"] = [
            asdict(m) for m in self._performance_metrics_buffer
        ]
        export_data["business_metrics"] = [
            asdict(m) for m in self._business_metrics_buffer
        ]

        # フィルターの適用
        if filters:
            for metric_type, metrics_list in export_data.items():
                export_data[metric_type] = self._apply_filters(metrics_list, filters)

        # エクスポート形式での出力
        if format == "json":
            return json.dumps(export_data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_collection_stats(self) -> dict[str, Any]:
        """収集統計情報取得"""
        return {
            "is_collecting": self._is_collecting,
            "buffer_sizes": {
                "search_metrics": len(self._search_metrics_buffer),
                "user_metrics": len(self._user_metrics_buffer),
                "system_metrics": len(self._system_metrics_buffer),
                "performance_metrics": len(self._performance_metrics_buffer),
                "business_metrics": len(self._business_metrics_buffer),
            },
            "active_sessions": len(self._active_sessions),
            "alert_states": len(self._alert_states),
            "cache_entries": len(self._aggregated_cache),
        }
