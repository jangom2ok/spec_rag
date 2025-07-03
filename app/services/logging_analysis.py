"""ログ収集・分析サービス

TDD実装：ログ収集・分析機能
- ログ収集: 構造化ログ、ログレベル管理、ローテーション
- ログ分析: パターン分析、異常検知、トレンド分析
- アラート: 閾値ベースアラート、異常検知アラート
- レポート: 定期レポート、カスタムレポート
- 統計: アクセス統計、エラー統計、パフォーマンス統計
"""

import asyncio
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """ログレベル"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def value_int(self) -> int:
        """数値表現を取得"""
        levels = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        return levels[self.value]


class LogFormat(str, Enum):
    """ログフォーマット"""

    JSON = "json"
    STRUCTURED = "structured"
    PLAIN = "plain"


class AnalysisType(str, Enum):
    """分析タイプ"""

    PATTERN_ANALYSIS = "pattern_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class TimeRange:
    """時間範囲"""

    start_time: datetime
    end_time: datetime

    def contains(self, timestamp: datetime) -> bool:
        """指定時刻が範囲内かチェック"""
        return self.start_time <= timestamp <= self.end_time

    def duration_minutes(self) -> float:
        """期間を分単位で取得"""
        return (self.end_time - self.start_time).total_seconds() / 60


@dataclass
class LogFilter:
    """ログフィルター"""

    field: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, not_in, contains


@dataclass
class AlertRule:
    """アラートルール"""

    name: str
    pattern: str
    threshold: int
    time_window: int  # 秒
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True


@dataclass
class LoggingConfig:
    """ログ設定"""

    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON

    # ローテーション設定
    enable_rotation: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_compression: bool = False

    # 分析設定
    enable_analysis: bool = False
    analysis_interval: int = 300  # 秒
    buffer_size: int = 10000

    # アラート設定
    enable_alerting: bool = False
    alert_rules: list[AlertRule] = field(default_factory=list)

    # 異常検知設定
    enable_anomaly_detection: bool = False
    anomaly_threshold: float = 2.0  # 標準偏差の倍数

    # 保持設定
    retention_days: int = 7
    cleanup_interval: int = 3600  # 秒

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be greater than 0")
        if self.backup_count <= 0:
            raise ValueError("backup_count must be greater than 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0")


@dataclass
class LogEntry:
    """ログエントリ"""

    timestamp: datetime
    level: LogLevel
    logger: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "logger": self.logger,
            "message": self.message,
            "context": self.context,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }

    def matches_filter(self, log_filter: LogFilter) -> bool:
        """フィルターにマッチするかチェック"""
        value = None

        if log_filter.field == "level":
            value = self.level.value
        elif log_filter.field == "logger":
            value = self.logger
        elif log_filter.field == "message":
            value = self.message
        elif log_filter.field in self.context:
            value = self.context[log_filter.field]
        else:
            return False

        if log_filter.operator == "eq":
            return value == log_filter.value
        elif log_filter.operator == "ne":
            return value != log_filter.value
        elif log_filter.operator == "gt":
            return value > log_filter.value
        elif log_filter.operator == "lt":
            return value < log_filter.value
        elif log_filter.operator == "gte":
            if log_filter.field == "level":
                return LogLevel(value).value_int >= LogLevel(log_filter.value).value_int
            return value >= log_filter.value
        elif log_filter.operator == "lte":
            if log_filter.field == "level":
                return LogLevel(value).value_int <= LogLevel(log_filter.value).value_int
            return value <= log_filter.value
        elif log_filter.operator == "in":
            return value in log_filter.value
        elif log_filter.operator == "not_in":
            return value not in log_filter.value
        elif log_filter.operator == "contains":
            return log_filter.value in str(value)

        return False


@dataclass
class PatternAnalysis:
    """パターン分析結果"""

    pattern_name: str
    pattern_type: str
    count: int
    percentage: float
    examples: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


@dataclass
class AnomalyDetection:
    """異常検知結果"""

    anomaly_type: str
    field: str
    value: float
    expected_range: tuple[float, float]
    severity: str
    timestamp: datetime
    description: str


@dataclass
class TrendAnalysis:
    """トレンド分析結果"""

    metric: str
    direction: str  # increasing, decreasing, stable
    confidence: float
    slope: float
    time_range: TimeRange
    data_points: int


@dataclass
class AlertEvent:
    """アラートイベント"""

    rule_name: str
    severity: str
    count: int
    threshold: int
    time_window: int
    triggered_at: datetime
    description: str


@dataclass
class LogStatistics:
    """ログ統計"""

    total_count: int
    time_range: TimeRange
    level_counts: dict[str, int]
    logger_counts: dict[str, int]
    error_rate: float
    average_per_minute: float


@dataclass
class LogAnalysisRequest:
    """ログ分析リクエスト"""

    analysis_types: list[AnalysisType]
    time_range: TimeRange
    filters: list[LogFilter] = field(default_factory=list)
    max_results: int = 1000
    include_examples: bool = True


@dataclass
class LogAnalysisResponse:
    """ログ分析レスポンス"""

    success: bool
    pattern_analyses: list[PatternAnalysis] = field(default_factory=list)
    anomaly_detections: list[AnomalyDetection] = field(default_factory=list)
    trend_analyses: list[TrendAnalysis] = field(default_factory=list)
    processing_time: float = 0.0
    analyzed_count: int = 0
    error_message: str | None = None


@dataclass
class LogReport:
    """ログレポート"""

    report_type: str
    generated_at: datetime
    time_range: TimeRange
    total_log_count: int
    error_summary: dict[str, Any]
    top_loggers: list[tuple[str, int]]
    pattern_analyses: list[PatternAnalysis] = field(default_factory=list)
    anomaly_detections: list[AnomalyDetection] = field(default_factory=list)
    trend_analyses: list[TrendAnalysis] = field(default_factory=list)


class LoggingAnalysisService:
    """ログ分析サービスメインクラス"""

    def __init__(self, config: LoggingConfig):
        self.config = config

        # 分析状態管理
        self._is_analyzing = False
        self._analysis_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # ログバッファ
        self._log_buffer: deque[LogEntry] = deque(maxlen=config.buffer_size)

        # 分析キャッシュ
        self._pattern_cache: dict[str, PatternAnalysis] = {}
        self._anomaly_cache: list[AnomalyDetection] = []
        self._trend_cache: list[TrendAnalysis] = []

        # アラート管理
        self._alert_states: dict[str, dict[str, Any]] = {}
        self._alert_callback: Callable[[dict[str, Any]], None] | None = None

        # 統計情報
        self._statistics_cache: dict[str, LogStatistics] = {}

    async def start_analysis(self) -> None:
        """ログ分析開始"""
        if self._is_analyzing:
            logger.warning("Log analysis is already running")
            return

        logger.info("Starting log analysis")
        self._is_analyzing = True

        # バックグラウンド分析スレッドの開始
        if self.config.enable_analysis:
            self._analysis_thread = threading.Thread(
                target=self._background_analysis_loop, daemon=True
            )
            self._analysis_thread.start()

    async def stop_analysis(self) -> None:
        """ログ分析停止"""
        if not self._is_analyzing:
            return

        logger.info("Stopping log analysis")
        self._is_analyzing = False
        self._stop_event.set()

        # スレッド終了待機
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5.0)

    def _background_analysis_loop(self) -> None:
        """バックグラウンド分析ループ"""
        while not self._stop_event.wait(self.config.analysis_interval):
            try:
                # 定期分析の実行
                asyncio.run_coroutine_threadsafe(
                    self._perform_periodic_analysis(), asyncio.get_event_loop()
                )

                # アラートチェック
                if self.config.enable_alerting:
                    asyncio.run_coroutine_threadsafe(
                        self._check_alerts(), asyncio.get_event_loop()
                    )

                # クリーンアップ
                asyncio.run_coroutine_threadsafe(
                    self._cleanup_old_data(), asyncio.get_event_loop()
                )

            except Exception as e:
                logger.error(f"Background analysis error: {e}")

    async def ingest_log_entry(self, entry: LogEntry) -> None:
        """ログエントリの取り込み"""
        try:
            # ログレベルフィルタリング
            if entry.level.value_int < self.config.log_level.value_int:
                return

            self._log_buffer.append(entry)

            # リアルタイムアラートチェック
            if self.config.enable_alerting:
                await self._check_real_time_alerts(entry)

        except Exception as e:
            logger.error(f"Failed to ingest log entry: {e}")

    async def analyze_logs(self, request: LogAnalysisRequest) -> LogAnalysisResponse:
        """ログ分析実行"""
        start_time = time.time()

        try:
            # 対象ログの抽出
            filtered_logs = self._filter_logs(request.time_range, request.filters)

            if not filtered_logs:
                return LogAnalysisResponse(
                    success=True,
                    analyzed_count=0,
                    processing_time=time.time() - start_time,
                )

            # 制限適用
            if len(filtered_logs) > request.max_results:
                filtered_logs = filtered_logs[: request.max_results]

            response = LogAnalysisResponse(
                success=True, analyzed_count=len(filtered_logs)
            )

            # 分析実行
            for analysis_type in request.analysis_types:
                if analysis_type == AnalysisType.PATTERN_ANALYSIS:
                    response.pattern_analyses = await self._analyze_patterns(
                        filtered_logs
                    )
                elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                    response.anomaly_detections = await self._detect_anomalies(
                        filtered_logs
                    )
                elif analysis_type == AnalysisType.TREND_ANALYSIS:
                    response.trend_analyses = await self._analyze_trends(filtered_logs)

            response.processing_time = time.time() - start_time
            return response

        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return LogAnalysisResponse(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def _filter_logs(
        self, time_range: TimeRange, filters: list[LogFilter]
    ) -> list[LogEntry]:
        """ログのフィルタリング"""
        filtered = []

        for entry in self._log_buffer:
            # 時間範囲チェック
            if not time_range.contains(entry.timestamp):
                continue

            # フィルターチェック
            if all(entry.matches_filter(f) for f in filters):
                filtered.append(entry)

        return filtered

    async def _analyze_patterns(self, logs: list[LogEntry]) -> list[PatternAnalysis]:
        """パターン分析"""
        if not logs:
            return []

        patterns = []

        # ログレベルパターン
        level_counts = Counter(entry.level.value for entry in logs)
        for level, count in level_counts.items():
            percentage = (count / len(logs)) * 100
            patterns.append(
                PatternAnalysis(
                    pattern_name=f"{level}_logs",
                    pattern_type="log_level",
                    count=count,
                    percentage=percentage,
                    examples=[
                        entry.message for entry in logs if entry.level.value == level
                    ][:3],
                    first_seen=min(
                        entry.timestamp for entry in logs if entry.level.value == level
                    ),
                    last_seen=max(
                        entry.timestamp for entry in logs if entry.level.value == level
                    ),
                )
            )

        # ロガーパターン
        logger_counts = Counter(entry.logger for entry in logs)
        for logger_name, count in logger_counts.most_common(10):
            percentage = (count / len(logs)) * 100
            patterns.append(
                PatternAnalysis(
                    pattern_name=f"logger_{logger_name}",
                    pattern_type="logger",
                    count=count,
                    percentage=percentage,
                    examples=[
                        entry.message for entry in logs if entry.logger == logger_name
                    ][:3],
                    first_seen=min(
                        entry.timestamp for entry in logs if entry.logger == logger_name
                    ),
                    last_seen=max(
                        entry.timestamp for entry in logs if entry.logger == logger_name
                    ),
                )
            )

        # エラーパターン検出
        error_logs = [
            entry for entry in logs if entry.level.value_int >= LogLevel.ERROR.value_int
        ]
        if error_logs:
            error_percentage = (len(error_logs) / len(logs)) * 100
            patterns.append(
                PatternAnalysis(
                    pattern_name="error_pattern",
                    pattern_type="error",
                    count=len(error_logs),
                    percentage=error_percentage,
                    examples=[entry.message for entry in error_logs][:5],
                    first_seen=min(entry.timestamp for entry in error_logs),
                    last_seen=max(entry.timestamp for entry in error_logs),
                )
            )

        return patterns

    async def _analyze_logger_frequency(self, logs: list[LogEntry]) -> dict[str, int]:
        """ロガー頻度分析"""
        return dict(Counter(entry.logger for entry in logs))

    async def _detect_anomalies(self, logs: list[LogEntry]) -> list[AnomalyDetection]:
        """異常検知"""
        anomalies = []

        if not logs or len(logs) < 10:
            return anomalies

        # レスポンス時間異常検知
        response_time_anomalies = await self._detect_response_time_anomalies(logs)
        anomalies.extend(response_time_anomalies)

        # エラー率異常検知
        error_rate_anomalies = await self._detect_error_rate_anomalies(logs)
        anomalies.extend(error_rate_anomalies)

        return anomalies

    async def _detect_response_time_anomalies(
        self, logs: list[LogEntry]
    ) -> list[AnomalyDetection]:
        """レスポンス時間異常検知"""
        anomalies = []

        # レスポンス時間データの抽出
        response_times = []
        for entry in logs:
            if "response_time" in entry.context:
                try:
                    response_times.append(float(entry.context["response_time"]))
                except (ValueError, TypeError):
                    continue

        if len(response_times) < 10:
            return anomalies

        # 統計計算
        mean_time = mean(response_times)
        std_time = stdev(response_times) if len(response_times) > 1 else 0

        if std_time == 0:
            return anomalies

        threshold = self.config.anomaly_threshold
        lower_bound = mean_time - (threshold * std_time)
        upper_bound = mean_time + (threshold * std_time)

        # 異常値検出
        for entry in logs:
            if "response_time" in entry.context:
                try:
                    response_time = float(entry.context["response_time"])
                    if response_time > upper_bound or response_time < lower_bound:
                        severity = (
                            "critical"
                            if response_time > mean_time + (3 * std_time)
                            else "warning"
                        )
                        anomalies.append(
                            AnomalyDetection(
                                anomaly_type="response_time",
                                field="response_time",
                                value=response_time,
                                expected_range=(lower_bound, upper_bound),
                                severity=severity,
                                timestamp=entry.timestamp,
                                description=f"Response time {response_time}ms is outside normal range {lower_bound:.1f}-{upper_bound:.1f}ms",
                            )
                        )
                except (ValueError, TypeError):
                    continue

        return anomalies

    async def _detect_error_rate_anomalies(
        self, logs: list[LogEntry]
    ) -> list[AnomalyDetection]:
        """エラー率異常検知"""
        anomalies = []

        if len(logs) < 20:
            return anomalies

        # 時間窓ごとのエラー率計算
        time_windows = self._group_logs_by_time_window(logs, minutes=5)
        error_rates = []

        for window_logs in time_windows.values():
            if len(window_logs) < 5:
                continue

            error_count = sum(
                1
                for entry in window_logs
                if entry.level.value_int >= LogLevel.ERROR.value_int
            )
            error_rate = error_count / len(window_logs)
            error_rates.append(error_rate)

        if len(error_rates) < 3:
            return anomalies

        # 異常なエラー率の検出
        mean_error_rate = mean(error_rates)
        std_error_rate = stdev(error_rates) if len(error_rates) > 1 else 0

        if std_error_rate == 0:
            return anomalies

        threshold = self.config.anomaly_threshold
        upper_bound = mean_error_rate + (threshold * std_error_rate)

        # 現在の時間窓でエラー率をチェック
        logs_list = list(logs)  # dequeをlistに変換
        recent_logs = logs_list[-min(20, len(logs_list)) :]  # 最新20件
        recent_error_count = sum(
            1
            for entry in recent_logs
            if entry.level.value_int >= LogLevel.ERROR.value_int
        )
        recent_error_rate = recent_error_count / len(recent_logs)

        if recent_error_rate > max(upper_bound, 0.3):  # 30%以上または統計的異常値
            severity = "critical" if recent_error_rate > 0.5 else "warning"
            anomalies.append(
                AnomalyDetection(
                    anomaly_type="error_rate",
                    field="error_rate",
                    value=recent_error_rate,
                    expected_range=(0.0, upper_bound),
                    severity=severity,
                    timestamp=logs[-1].timestamp,
                    description=f"Error rate {recent_error_rate:.2%} is significantly higher than normal ({mean_error_rate:.2%})",
                )
            )

        return anomalies

    def _group_logs_by_time_window(
        self, logs: list[LogEntry], minutes: int
    ) -> dict[str, list[LogEntry]]:
        """時間窓でログをグループ化"""
        groups = defaultdict(list)

        for entry in logs:
            # 分単位で丸める
            window_start = entry.timestamp.replace(second=0, microsecond=0)
            window_start = window_start.replace(
                minute=(window_start.minute // minutes) * minutes
            )
            window_key = window_start.strftime("%Y-%m-%d %H:%M")
            groups[window_key].append(entry)

        return groups

    async def _analyze_trends(self, logs: list[LogEntry]) -> list[TrendAnalysis]:
        """トレンド分析"""
        trends = []

        if len(logs) < 10:
            return trends

        # ログ量トレンド
        volume_trend = await self._analyze_volume_trend(logs)
        if volume_trend:
            trends.append(volume_trend)

        # エラー率トレンド
        error_rate_trends = await self._analyze_error_rate_trends(logs)
        trends.extend(error_rate_trends)

        return trends

    async def _analyze_volume_trend(self, logs: list[LogEntry]) -> TrendAnalysis | None:
        """ログ量トレンド分析"""
        time_windows = self._group_logs_by_time_window(logs, minutes=10)

        if len(time_windows) < 3:
            return None

        # 時系列データの準備
        sorted_windows = sorted(time_windows.items())
        volumes = [len(window_logs) for _, window_logs in sorted_windows]

        # 線形回帰で傾きを計算
        n = len(volumes)
        x_values = list(range(n))

        if n < 2:
            return None

        x_mean = mean(x_values)
        y_mean = mean(volumes)

        numerator = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_values, volumes, strict=False)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return None

        slope = numerator / denominator

        # トレンド方向の判定
        direction = "stable"
        confidence = 0.0

        if abs(slope) > 0.5:
            direction = "increasing" if slope > 0 else "decreasing"
            # 信頼度の簡易計算
            variance = (
                sum(
                    (volumes[i] - (y_mean + slope * (i - x_mean))) ** 2
                    for i in range(n)
                )
                / n
            )
            confidence = max(
                0.0, min(1.0, 1.0 - (variance / (y_mean**2 if y_mean > 0 else 1)))
            )

        time_range = TimeRange(
            start_time=logs[0].timestamp, end_time=logs[-1].timestamp
        )

        return TrendAnalysis(
            metric="log_volume",
            direction=direction,
            confidence=confidence,
            slope=slope,
            time_range=time_range,
            data_points=n,
        )

    async def _analyze_error_rate_trends(
        self, logs: list[LogEntry]
    ) -> list[TrendAnalysis]:
        """エラー率トレンド分析"""
        trends = []
        time_windows = self._group_logs_by_time_window(logs, minutes=10)

        if len(time_windows) < 3:
            return trends

        # エラー率時系列データの準備
        sorted_windows = sorted(time_windows.items())
        error_rates = []

        for _, window_logs in sorted_windows:
            if len(window_logs) == 0:
                continue
            error_count = sum(
                1
                for entry in window_logs
                if entry.level.value_int >= LogLevel.ERROR.value_int
            )
            error_rate = error_count / len(window_logs)
            error_rates.append(error_rate)

        if len(error_rates) < 3:
            return trends

        # 線形回帰
        n = len(error_rates)
        x_values = list(range(n))
        x_mean = mean(x_values)
        y_mean = mean(error_rates)

        numerator = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(x_values, error_rates, strict=False)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return trends

        slope = numerator / denominator

        # トレンド判定
        direction = "stable"
        confidence = 0.0

        if abs(slope) > 0.01:  # 1%の変化を閾値とする
            direction = "increasing" if slope > 0 else "decreasing"
            variance = (
                sum(
                    (error_rates[i] - (y_mean + slope * (i - x_mean))) ** 2
                    for i in range(n)
                )
                / n
            )
            confidence = max(
                0.0, min(1.0, 1.0 - (variance / (y_mean**2 if y_mean > 0 else 1)))
            )

        time_range = TimeRange(
            start_time=logs[0].timestamp, end_time=logs[-1].timestamp
        )

        trends.append(
            TrendAnalysis(
                metric="error_rate",
                direction=direction,
                confidence=confidence,
                slope=slope,
                time_range=time_range,
                data_points=n,
            )
        )

        return trends

    async def _check_alerts(self) -> None:
        """アラートチェック"""
        try:
            recent_logs = list(self._log_buffer)[-1000:]  # 最新1000件
            alerts = await self._evaluate_alert_rules(recent_logs)

            for alert in alerts:
                await self._trigger_alert(alert)

        except Exception as e:
            logger.error(f"Alert check failed: {e}")

    async def _check_real_time_alerts(self, entry: LogEntry) -> None:
        """リアルタイムアラートチェック"""
        try:
            # クリティカルログの即座アラート
            if entry.level == LogLevel.CRITICAL:
                alert = AlertEvent(
                    rule_name="critical_errors",
                    severity="critical",
                    count=1,
                    threshold=1,
                    time_window=60,
                    triggered_at=entry.timestamp,
                    description=f"Critical error detected: {entry.message}",
                )
                await self._trigger_alert(alert)

        except Exception as e:
            logger.error(f"Real-time alert check failed: {e}")

    async def _evaluate_alert_rules(self, logs: list[LogEntry]) -> list[AlertEvent]:
        """アラートルール評価"""
        alerts = []

        for rule in self.config.alert_rules:
            if not rule.enabled:
                continue

            try:
                alert = await self._evaluate_single_rule(rule, logs)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.name}: {e}")

        return alerts

    async def _evaluate_single_rule(
        self, rule: AlertRule, logs: list[LogEntry]
    ) -> AlertEvent | None:
        """単一アラートルールの評価"""
        cutoff_time = datetime.now() - timedelta(seconds=rule.time_window)
        if logs:
            # 最新ログのタイムスタンプを基準にする
            latest_time = max(log.timestamp for log in logs)
            cutoff_time = latest_time - timedelta(seconds=rule.time_window)
        recent_logs = [log for log in logs if log.timestamp >= cutoff_time]

        # パターンマッチング
        matching_count = 0

        if rule.pattern.upper() in ["ERROR", "WARNING", "CRITICAL", "INFO", "DEBUG"]:
            # ログレベルパターン
            target_level = LogLevel(rule.pattern.upper())
            matching_count = sum(1 for log in recent_logs if log.level == target_level)
        else:
            # メッセージパターン
            matching_count = sum(
                1 for log in recent_logs if rule.pattern.lower() in log.message.lower()
            )

        if matching_count >= rule.threshold:
            # アラート重複チェック
            alert_key = f"{rule.name}_{rule.severity}"
            if alert_key in self._alert_states:
                last_alert = self._alert_states[alert_key]
                if (
                    datetime.now() - last_alert["triggered_at"]
                ).total_seconds() < 300:  # 5分間隔
                    return None

            alert = AlertEvent(
                rule_name=rule.name,
                severity=rule.severity,
                count=matching_count,
                threshold=rule.threshold,
                time_window=rule.time_window,
                triggered_at=datetime.now(),
                description=f"Rule '{rule.name}' triggered: {matching_count} matches in {rule.time_window}s (threshold: {rule.threshold})",
            )

            self._alert_states[alert_key] = {
                "triggered_at": alert.triggered_at,
                "count": matching_count,
            }

            return alert

        return None

    async def _trigger_alert(self, alert: AlertEvent) -> None:
        """アラート発動"""
        try:
            logger.warning(f"ALERT: {alert.description}")

            if self._alert_callback:
                await self._alert_callback(
                    {
                        "rule_name": alert.rule_name,
                        "severity": alert.severity,
                        "count": alert.count,
                        "threshold": alert.threshold,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "description": alert.description,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    def set_alert_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """アラートコールバック設定"""
        self._alert_callback = callback

    async def get_log_statistics(
        self, start_time: datetime, end_time: datetime
    ) -> LogStatistics:
        """ログ統計取得"""
        time_range = TimeRange(start_time, end_time)
        filtered_logs = self._filter_logs(time_range, [])

        if not filtered_logs:
            return LogStatistics(
                total_count=0,
                time_range=time_range,
                level_counts={},
                logger_counts={},
                error_rate=0.0,
                average_per_minute=0.0,
            )

        # 統計計算
        level_counts = dict(Counter(entry.level.value for entry in filtered_logs))
        logger_counts = dict(Counter(entry.logger for entry in filtered_logs))

        error_count = sum(
            1
            for entry in filtered_logs
            if entry.level.value_int >= LogLevel.ERROR.value_int
        )
        error_rate = error_count / len(filtered_logs)

        duration_minutes = time_range.duration_minutes()
        average_per_minute = len(filtered_logs) / max(duration_minutes, 1)

        return LogStatistics(
            total_count=len(filtered_logs),
            time_range=time_range,
            level_counts=level_counts,
            logger_counts=logger_counts,
            error_rate=error_rate,
            average_per_minute=average_per_minute,
        )

    async def generate_daily_report(self, report_date: date) -> LogReport:
        """日次レポート生成"""
        start_time = datetime.combine(report_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        logs = self._filter_logs(time_range, [])
        stats = await self.get_log_statistics(start_time, end_time)

        # エラーサマリー
        error_logs = [
            log for log in logs if log.level.value_int >= LogLevel.ERROR.value_int
        ]
        error_summary = {
            "total_errors": len(error_logs),
            "error_rate": stats.error_rate,
            "top_error_loggers": dict(
                Counter(log.logger for log in error_logs).most_common(5)
            ),
        }

        # トップロガー
        top_loggers = list(Counter(log.logger for log in logs).most_common(10))

        return LogReport(
            report_type="daily",
            generated_at=datetime.now(),
            time_range=time_range,
            total_log_count=len(logs),
            error_summary=error_summary,
            top_loggers=top_loggers,
        )

    async def generate_custom_report(self, report_config: dict[str, Any]) -> LogReport:
        """カスタムレポート生成"""
        time_range = report_config["time_range"]
        logs = self._filter_logs(time_range, [])

        report = LogReport(
            report_type="custom",
            generated_at=datetime.now(),
            time_range=time_range,
            total_log_count=len(logs),
            error_summary={},
            top_loggers=[],
        )

        # 設定に基づく分析実行
        if report_config.get("include_patterns", False):
            report.pattern_analyses = await self._analyze_patterns(logs)

        if report_config.get("include_anomalies", False):
            report.anomaly_detections = await self._detect_anomalies(logs)

        if report_config.get("include_trends", False):
            report.trend_analyses = await self._analyze_trends(logs)

        return report

    async def _perform_periodic_analysis(self) -> None:
        """定期分析処理"""
        try:
            recent_logs = list(self._log_buffer)[-1000:]

            # パターン分析更新
            patterns = await self._analyze_patterns(recent_logs)
            for pattern in patterns:
                self._pattern_cache[pattern.pattern_name] = pattern

            # 異常検知実行
            if self.config.enable_anomaly_detection:
                anomalies = await self._detect_anomalies(recent_logs)
                self._anomaly_cache.extend(anomalies)
                # 古い異常データのクリーンアップ
                cutoff_time = datetime.now() - timedelta(hours=24)
                self._anomaly_cache = [
                    a for a in self._anomaly_cache if a.timestamp >= cutoff_time
                ]

        except Exception as e:
            logger.error(f"Periodic analysis failed: {e}")

    async def _cleanup_old_data(self) -> None:
        """古いデータのクリーンアップ"""
        try:
            # cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)

            # ログバッファのクリーンアップは自動（dequeのmaxlen）

            # キャッシュのクリーンアップ
            self._pattern_cache.clear()  # パターンキャッシュは定期的にクリア

            # アラート状態の古いエントリをクリーンアップ
            old_alert_keys = []
            for key, state in self._alert_states.items():
                if (
                    datetime.now() - state["triggered_at"]
                ).total_seconds() > 3600:  # 1時間
                    old_alert_keys.append(key)

            for key in old_alert_keys:
                del self._alert_states[key]

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def get_analysis_status(self) -> dict[str, Any]:
        """分析状況取得"""
        return {
            "is_analyzing": self._is_analyzing,
            "buffer_size": len(self._log_buffer),
            "pattern_cache_size": len(self._pattern_cache),
            "anomaly_cache_size": len(self._anomaly_cache),
            "alert_states": len(self._alert_states),
            "config": asdict(self.config),
        }
