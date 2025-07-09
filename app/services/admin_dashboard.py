"""管理画面サービス

TDD実装：検索分析・システム状態管理ダッシュボード
- 検索分析: クエリ統計、パフォーマンス分析、ユーザー行動分析
- システム状態: リアルタイム監視、リソース使用状況、アラート状況
- ダッシュボード: カスタマイズ可能なウィジェット、レポート生成
- ユーザー管理: 権限管理、アクセス制御、監査ログ
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


class WidgetType(str, Enum):
    """ウィジェットタイプ"""

    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    AREA_CHART = "area_chart"
    GAUGE = "gauge"
    NUMBER = "number"
    TABLE = "table"
    MAP = "map"


class ReportType(str, Enum):
    """レポートタイプ"""

    SYSTEM_STATUS = "system_status"
    SEARCH_ANALYTICS = "search_analytics"
    USER_ANALYTICS = "user_analytics"
    DASHBOARD_SUMMARY = "dashboard_summary"
    PERFORMANCE_REPORT = "performance_report"
    SECURITY_AUDIT = "security_audit"


class ReportFormat(str, Enum):
    """レポート形式"""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""

    enable_real_time: bool = True
    refresh_interval: int = 60  # 秒
    max_widgets: int = 10
    default_theme: str = "light"
    enable_export: bool = True
    enable_notifications: bool = True

    # 高度な機能
    enable_user_management: bool = False
    enable_audit_log: bool = False
    enable_custom_reports: bool = False
    data_retention_days: int = 30
    cache_duration: int = 300  # 秒
    max_concurrent_users: int = 50

    # パフォーマンス設定
    batch_size: int = 100
    query_timeout: int = 30  # 秒
    max_memory_usage: int = 1024 * 1024 * 512  # 512MB

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be greater than 0")
        if self.max_widgets <= 0:
            raise ValueError("max_widgets must be greater than 0")
        if self.data_retention_days <= 0:
            raise ValueError("data_retention_days must be greater than 0")


@dataclass
class DashboardWidget:
    """ダッシュボードウィジェット"""

    id: str
    title: str
    type: WidgetType
    config: dict[str, Any]
    position: dict[str, int]  # x, y, width, height
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def validate_position(self) -> bool:
        """位置設定の検証"""
        required_fields = ["x", "y", "width", "height"]
        return all(field in self.position for field in required_fields)

    def validate_config(self) -> bool:
        """設定の検証"""
        if self.type in [
            WidgetType.LINE_CHART,
            WidgetType.BAR_CHART,
            WidgetType.AREA_CHART,
        ]:
            return "metrics" in self.config
        elif self.type == WidgetType.GAUGE:
            return "metrics" in self.config and "thresholds" in self.config
        elif self.type == WidgetType.TABLE:
            return "columns" in self.config
        return True

    def get_required_data_fields(self) -> list[str]:
        """必要データフィールドの取得"""
        if "metrics" in self.config:
            return self.config["metrics"]
        elif "columns" in self.config:
            return self.config["columns"]
        return []

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: datetime
    cpu_usage: float  # 0.0-1.0
    memory_usage: float  # 0.0-1.0
    disk_usage: float  # 0.0-1.0
    network_io: dict[str, int]  # bytes_sent, bytes_recv
    active_connections: int
    response_time: float  # ms
    error_rate: float  # 0.0-1.0
    throughput: float  # requests/sec

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class SearchAnalytics:
    """検索分析データ"""

    total_searches: int
    successful_searches: int
    failed_searches: int
    avg_response_time: float
    success_rate: float
    popular_queries: list[dict[str, Any]]
    search_trends: dict[str, list[float]]
    user_search_patterns: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class UserAnalytics:
    """ユーザー分析データ"""

    active_users: int
    total_sessions: int
    avg_session_duration: float
    bounce_rate: float
    top_users: list[dict[str, Any]]
    user_activity_timeline: dict[str, list[dict[str, Any]]]
    geographic_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class ReportConfig:
    """レポート設定"""

    type: ReportType
    format: ReportFormat
    time_range: timedelta
    include_charts: bool = True
    include_raw_data: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    aggregation: str = "hour"  # hour, day, week, month

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result["time_range"] = self.time_range.total_seconds()
        return result


@dataclass
class Report:
    """生成されたレポート"""

    id: str
    type: ReportType
    format: ReportFormat
    generated_at: datetime
    content: bytes
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換（コンテンツ除く）"""
        result = asdict(self)
        result.pop("content")  # バイナリコンテンツは除外
        return result


class AdminDashboard:
    """管理ダッシュボードサービスメインクラス"""

    def __init__(self, config: DashboardConfig):
        self.config = config

        # ダッシュボード状態管理
        self._widgets: dict[str, DashboardWidget] = {}
        self._is_real_time_enabled = config.enable_real_time
        self._monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # データストレージ
        self._search_events: deque[dict[str, Any]] = deque(maxlen=10000)
        self._user_activities: deque[dict[str, Any]] = deque(maxlen=10000)
        self._system_metrics_history: deque[SystemMetrics] = deque(maxlen=1000)
        self._audit_logs: deque[dict[str, Any]] = deque(maxlen=5000)

        # キャッシュ
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # ユーザー管理
        self._user_permissions: dict[str, dict[str, bool]] = {}
        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._scheduled_reports: list[dict[str, Any]] = []

        # リアルタイムコールバック
        self._real_time_callback: Callable[[dict[str, Any]], None] | None = None

        # パフォーマンス監視
        self._performance_metrics: dict[str, list[float]] = defaultdict(list)

    async def add_widget(self, widget: DashboardWidget) -> None:
        """ウィジェット追加"""
        if len(self._widgets) >= self.config.max_widgets:
            raise ValueError(
                f"Maximum widgets limit reached: {self.config.max_widgets}"
            )

        if not widget.validate_position() or not widget.validate_config():
            raise ValueError("Invalid widget configuration")

        widget.created_at = datetime.now()
        widget.updated_at = datetime.now()
        self._widgets[widget.id] = widget
        logger.info(f"Added widget: {widget.id}")

    async def remove_widget(self, widget_id: str) -> None:
        """ウィジェット削除"""
        if widget_id in self._widgets:
            del self._widgets[widget_id]
            logger.info(f"Removed widget: {widget_id}")

    async def update_widget_config(
        self, widget_id: str, new_config: dict[str, Any]
    ) -> None:
        """ウィジェット設定更新"""
        if widget_id not in self._widgets:
            raise ValueError(f"Widget not found: {widget_id}")

        widget = self._widgets[widget_id]
        widget.config.update(new_config)
        widget.updated_at = datetime.now()
        logger.info(f"Updated widget config: {widget_id}")

    async def collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # ディスク使用率
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # ネットワークI/O
            network_io = psutil.net_io_counters()

            # モックデータ（実際の実装では実際のメトリクスを使用）
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent / 100.0,
                memory_usage=memory_percent / 100.0,
                disk_usage=disk_percent / 100.0,
                network_io={
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                },
                active_connections=len(self._active_sessions),
                response_time=150.0,  # モック値
                error_rate=0.02,  # モック値
                throughput=100.0,  # モック値
            )

            self._system_metrics_history.append(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # フォールバックメトリクス
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={"bytes_sent": 0, "bytes_recv": 0},
                active_connections=0,
                response_time=0.0,
                error_rate=0.0,
                throughput=0.0,
            )

    async def record_search_event(self, event: dict[str, Any]) -> None:
        """検索イベント記録"""
        event["recorded_at"] = datetime.now()
        self._search_events.append(event)

    async def record_user_activity(self, activity: dict[str, Any]) -> None:
        """ユーザー活動記録"""
        activity["recorded_at"] = datetime.now()
        self._user_activities.append(activity)

    async def get_search_analytics(
        self, start_time: datetime, end_time: datetime
    ) -> SearchAnalytics:
        """検索分析データ取得"""
        # キャッシュキーの生成
        cache_key = f"search_analytics_{start_time.isoformat()}_{end_time.isoformat()}"

        # キャッシュチェック
        if self._is_cache_valid(cache_key):
            cached_data = self._cache[cache_key]
            return SearchAnalytics(**cached_data)

        # 指定期間内の検索イベントを収集
        relevant_events = [
            event
            for event in self._search_events
            if start_time <= event.get("timestamp", datetime.now()) <= end_time
        ]

        total_searches = len(relevant_events)
        successful_searches = len(
            [e for e in relevant_events if e.get("success", False)]
        )
        failed_searches = total_searches - successful_searches

        avg_response_time = 0.0
        if relevant_events:
            response_times = [e.get("response_time", 0) for e in relevant_events]
            avg_response_time = mean(response_times)

        success_rate = (
            successful_searches / total_searches if total_searches > 0 else 0.0
        )

        # 人気クエリの集計
        query_counts: dict[str, int] = defaultdict(int)
        for event in relevant_events:
            query = event.get("query", "")
            if query:
                query_counts[query] += 1

        popular_queries = [
            {"query": query, "count": count}
            for query, count in sorted(
                query_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        analytics = SearchAnalytics(
            total_searches=total_searches,
            successful_searches=successful_searches,
            failed_searches=failed_searches,
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            popular_queries=popular_queries,
            search_trends={},  # 実装省略
            user_search_patterns={},  # 実装省略
        )

        # キャッシュに保存
        self._cache[cache_key] = analytics.to_dict()
        self._cache_timestamps[cache_key] = datetime.now()

        return analytics

    async def get_user_analytics(
        self, start_time: datetime, end_time: datetime
    ) -> UserAnalytics:
        """ユーザー分析データ取得"""
        # 指定期間内のユーザー活動を収集
        relevant_activities = [
            activity
            for activity in self._user_activities
            if start_time <= activity.get("timestamp", datetime.now()) <= end_time
        ]

        # ユニークユーザー数
        active_users = len(
            {activity.get("user_id") for activity in relevant_activities}
        )

        # セッション分析
        sessions = {activity.get("session_id") for activity in relevant_activities}
        total_sessions = len(sessions)

        analytics = UserAnalytics(
            active_users=active_users,
            total_sessions=total_sessions,
            avg_session_duration=1800.0,  # モック値: 30分
            bounce_rate=0.25,  # モック値
            top_users=[],  # 実装省略
            user_activity_timeline={},  # 実装省略
            geographic_distribution={},  # 実装省略
        )

        return analytics

    async def generate_report(self, report_config: ReportConfig) -> Report:
        """レポート生成"""
        report_id = f"report_{int(time.time())}"
        end_time = datetime.now()
        start_time = end_time - report_config.time_range

        try:
            if report_config.type == ReportType.SYSTEM_STATUS:
                content = await self._generate_system_status_report(
                    start_time, end_time, report_config
                )
            elif report_config.type == ReportType.SEARCH_ANALYTICS:
                content = await self._generate_search_analytics_report(
                    start_time, end_time, report_config
                )
            elif report_config.type == ReportType.USER_ANALYTICS:
                content = await self._generate_user_analytics_report(
                    start_time, end_time, report_config
                )
            elif report_config.type == ReportType.DASHBOARD_SUMMARY:
                content = await self._generate_dashboard_summary_report(
                    start_time, end_time, report_config
                )
            else:
                raise ValueError(f"Unsupported report type: {report_config.type}")

            report = Report(
                id=report_id,
                type=report_config.type,
                format=report_config.format,
                generated_at=datetime.now(),
                content=content,
                size_bytes=len(content),
                metadata={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "config": report_config.to_dict(),
                },
            )

            logger.info(f"Generated report: {report_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise

    async def _generate_system_status_report(
        self, start_time: datetime, end_time: datetime, config: ReportConfig
    ) -> bytes:
        """システム状態レポート生成"""
        metrics = await self.collect_system_metrics()

        if config.format == ReportFormat.JSON:
            data = {
                "report_type": "system_status",
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "metrics": metrics.to_dict(),
                "widgets_count": len(self._widgets),
                "active_sessions": len(self._active_sessions),
            }
            return json.dumps(data, indent=2).encode("utf-8")
        elif config.format == ReportFormat.CSV:
            # CSV形式の実装（簡略版）
            csv_content = "timestamp,cpu_usage,memory_usage,disk_usage\n"
            csv_content += f"{metrics.timestamp.isoformat()},{metrics.cpu_usage},{metrics.memory_usage},{metrics.disk_usage}\n"
            return csv_content.encode("utf-8")
        else:
            # PDF形式の実装（モック）
            return b"PDF content placeholder"

    async def _generate_search_analytics_report(
        self, start_time: datetime, end_time: datetime, config: ReportConfig
    ) -> bytes:
        """検索分析レポート生成"""
        analytics = await self.get_search_analytics(start_time, end_time)

        if config.format == ReportFormat.JSON:
            data = {
                "report_type": "search_analytics",
                "generated_at": datetime.now().isoformat(),
                "analytics": analytics.to_dict(),
            }
            return json.dumps(data, indent=2).encode("utf-8")
        elif config.format == ReportFormat.CSV:
            csv_content = "metric,value\n"
            csv_content += f"total_searches,{analytics.total_searches}\n"
            csv_content += f"success_rate,{analytics.success_rate}\n"
            csv_content += f"avg_response_time,{analytics.avg_response_time}\n"
            return csv_content.encode("utf-8")
        else:
            return b"PDF content placeholder"

    async def _generate_user_analytics_report(
        self, start_time: datetime, end_time: datetime, config: ReportConfig
    ) -> bytes:
        """ユーザー分析レポート生成"""
        analytics = await self.get_user_analytics(start_time, end_time)

        data = {
            "report_type": "user_analytics",
            "generated_at": datetime.now().isoformat(),
            "analytics": analytics.to_dict(),
        }
        return json.dumps(data, indent=2).encode("utf-8")

    async def _generate_dashboard_summary_report(
        self, start_time: datetime, end_time: datetime, config: ReportConfig
    ) -> bytes:
        """ダッシュボードサマリーレポート生成"""
        status = await self.get_dashboard_status()

        data = {
            "report_type": "dashboard_summary",
            "generated_at": datetime.now().isoformat(),
            "dashboard_status": status,
        }
        return json.dumps(data, indent=2).encode("utf-8")

    async def schedule_report(self, schedule_config: dict[str, Any]) -> None:
        """定期レポートのスケジュール設定"""
        schedule_config["id"] = f"schedule_{int(time.time())}"
        schedule_config["created_at"] = datetime.now().isoformat()
        self._scheduled_reports.append(schedule_config)
        logger.info(f"Scheduled report: {schedule_config['id']}")

    async def get_scheduled_reports(self) -> list[dict[str, Any]]:
        """定期レポート一覧取得"""
        return self._scheduled_reports.copy()

    async def start_real_time_monitoring(self) -> None:
        """リアルタイム監視開始"""
        if self._monitoring_thread is not None:
            logger.warning("Real-time monitoring is already running")
            return

        logger.info("Starting real-time monitoring")
        self._is_real_time_enabled = True
        self._stop_event.clear()

        self._monitoring_thread = threading.Thread(
            target=self._real_time_monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    async def stop_real_time_monitoring(self) -> None:
        """リアルタイム監視停止"""
        if not self._is_real_time_enabled:
            return

        logger.info("Stopping real-time monitoring")
        self._is_real_time_enabled = False
        self._stop_event.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        self._monitoring_thread = None

    def _real_time_monitoring_loop(self) -> None:
        """リアルタイム監視ループ"""
        while not self._stop_event.wait(self.config.refresh_interval):
            try:
                # メトリクス収集
                metrics = asyncio.run_coroutine_threadsafe(
                    self.collect_system_metrics(), asyncio.get_event_loop()
                ).result()

                # リアルタイムコールバック実行
                if self._real_time_callback:
                    try:
                        self._real_time_callback(metrics.to_dict())
                    except Exception as e:
                        logger.error(f"Real-time callback error: {e}")

            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")

    def set_real_time_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """リアルタイムコールバック設定"""
        self._real_time_callback = callback

    async def set_user_permissions(
        self, user_id: str, permissions: dict[str, bool]
    ) -> None:
        """ユーザー権限設定"""
        self._user_permissions[user_id] = permissions
        logger.info(f"Set permissions for user: {user_id}")

    async def get_user_permissions(self, user_id: str) -> dict[str, bool]:
        """ユーザー権限取得"""
        return self._user_permissions.get(user_id, {})

    async def log_user_action(self, action_log: dict[str, Any]) -> None:
        """ユーザー操作ログ記録"""
        action_log["logged_at"] = datetime.now()
        self._audit_logs.append(action_log)

    async def get_audit_logs(
        self, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """監査ログ取得"""
        return [
            log
            for log in self._audit_logs
            if start_time <= log.get("timestamp", datetime.now()) <= end_time
        ]

    async def track_user_session(self, session_info: dict[str, Any]) -> None:
        """ユーザーセッション追跡"""
        session_id = session_info["session_id"]
        self._active_sessions[session_id] = session_info
        logger.info(f"Tracking session: {session_id}")

    async def get_active_sessions(self) -> list[dict[str, Any]]:
        """アクティブセッション取得"""
        return list(self._active_sessions.values())

    async def get_dashboard_status(self) -> dict[str, Any]:
        """ダッシュボード状況取得"""
        return {
            "widget_count": len(self._widgets),
            "real_time_enabled": self._is_real_time_enabled,
            "active_sessions": len(self._active_sessions),
            "search_events_count": len(self._search_events),
            "user_activities_count": len(self._user_activities),
            "system_metrics_count": len(self._system_metrics_history),
            "audit_logs_count": len(self._audit_logs),
            "config": asdict(self.config),
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュの有効性チェック"""
        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        return datetime.now() - cache_time < timedelta(
            seconds=self.config.cache_duration
        )
