"""アラート機能サービス

TDD実装：パフォーマンス・エラー監視アラート機能
- しきい値ベースアラート: レスポンス時間、エラー率、CPU/メモリ使用率
- 複合条件アラート: 複数メトリクスの組み合わせ
- アラートエスカレーション: 重要度別の通知先設定
- 抑制・統合機能: 類似アラートの重複抑制
- 通知チャネル: Email、Slack、Webhook統合
"""

import asyncio
import logging
import smtplib
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """アラート重要度"""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertStatus(str, Enum):
    """アラート状態"""

    FIRING = "FIRING"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"


class ComparisonOperator(str, Enum):
    """比較演算子"""

    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"


class AlertChannel(str, Enum):
    """アラートチャネル"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertCondition:
    """アラート条件"""

    operator: ComparisonOperator
    threshold: float
    duration: int  # 秒

    def evaluate(self, value: float) -> bool:
        """条件評価"""
        if self.operator == ComparisonOperator.GREATER_THAN:
            return value > self.threshold
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            return value < self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= self.threshold
        elif self.operator == ComparisonOperator.EQUAL:
            return value == self.threshold
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return value != self.threshold
        
        return False  # type: ignore[unreachable]


@dataclass
class AlertRule:
    """アラートルール"""

    id: str
    name: str
    description: str
    metric: str
    condition: AlertCondition
    severity: AlertSeverity
    enabled: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    cooldown: int = 300  # 5分間のクールダウン

    def matches_metric(self, metric_name: str) -> bool:
        """メトリクス名にマッチするかチェック"""
        return self.metric == metric_name and self.enabled


@dataclass
class NotificationChannel:
    """通知チャネル"""

    id: str
    name: str
    type: AlertChannel
    config: dict[str, Any]
    enabled: bool = True
    severity_filter: list[AlertSeverity] = field(default_factory=list)

    def can_send(self, severity: AlertSeverity) -> bool:
        """指定重要度の通知を送信可能かチェック"""
        if not self.enabled:
            return False
        if not self.severity_filter:
            return True
        return severity in self.severity_filter


@dataclass
class EscalationPolicy:
    """エスカレーションポリシー"""

    id: str
    name: str
    rules: list[dict[str, Any]]
    enabled: bool = True

    def get_escalation_steps(self, severity: AlertSeverity) -> list[dict[str, Any]]:
        """指定重要度のエスカレーション手順を取得"""
        if not self.enabled:
            return []

        return [
            rule for rule in self.rules
            if rule.get("severity") == severity.value
        ]


@dataclass
class SuppressRule:
    """抑制ルール"""

    id: str
    name: str
    conditions: list[dict[str, Any]]
    duration: int  # 秒
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        """抑制期間が期限切れかチェック"""
        return datetime.now() > self.created_at + timedelta(seconds=self.duration)

    def matches(self, metric: str, value: float) -> bool:
        """抑制条件にマッチするかチェック"""
        if not self.enabled or self.is_expired():
            return False

        for condition in self.conditions:
            if condition["metric"] == metric:
                operator = condition["operator"]
                threshold = condition["value"]

                if operator == "gt" and value > threshold:
                    return True
                elif operator == "gte" and value >= threshold:
                    return True
                elif operator == "lt" and value < threshold:
                    return True
                elif operator == "lte" and value <= threshold:
                    return True
                elif operator == "eq" and value == threshold:
                    return True

        return False


@dataclass
class AlertTriggerEvent:
    """アラート発動イベント"""

    rule_id: str
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    description: str
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class AlertingConfig:
    """アラート設定"""

    enable_alerting: bool = True
    evaluation_interval: int = 60  # 秒
    default_severity: AlertSeverity = AlertSeverity.WARNING
    max_alerts_per_minute: int = 10
    alert_retention_days: int = 30

    # 高度な機能
    enable_escalation: bool = False
    enable_suppression: bool = False
    enable_grouping: bool = False
    grouping_window: int = 300  # 秒
    notification_cooldown: int = 300  # 秒

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be greater than 0")
        if self.alert_retention_days <= 0:
            raise ValueError("alert_retention_days must be greater than 0")
        if self.max_alerts_per_minute <= 0:
            raise ValueError("max_alerts_per_minute must be greater than 0")


class AlertingService:
    """アラート機能サービスメインクラス"""

    def __init__(self, config: AlertingConfig):
        self.config = config

        # 監視状態管理
        self._is_monitoring = False
        self._monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # アラートルール管理
        self._alert_rules: dict[str, AlertRule] = {}
        self._notification_channels: dict[str, NotificationChannel] = {}
        self._escalation_policies: dict[str, EscalationPolicy] = {}
        self._suppress_rules: dict[str, SuppressRule] = {}

        # アラート状態管理
        self._active_alerts: dict[str, list[AlertTriggerEvent]] = defaultdict(list)
        self._alert_history: deque[AlertTriggerEvent] = deque(maxlen=10000)
        self._last_alert_times: dict[str, datetime] = {}

        # グルーピング・抑制管理
        self._alert_groups: dict[str, list[AlertTriggerEvent]] = defaultdict(list)
        self._escalation_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._active_suppressions: set[str] = set()

        # 通知履歴
        self._notification_history: deque[dict[str, Any]] = deque(maxlen=1000)

        # アラートコールバック
        self._alert_callback: Callable[[dict[str, Any]], None] | None = None

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_monitoring:
            logger.warning("Alerting monitoring is already running")
            return

        logger.info("Starting alerting monitoring")
        self._is_monitoring = True

        # バックグラウンド監視スレッドの開始
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    async def stop_monitoring(self) -> None:
        """監視停止"""
        if not self._is_monitoring:
            return

        logger.info("Stopping alerting monitoring")
        self._is_monitoring = False
        self._stop_event.set()

        # スレッド終了待機
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

    def _background_monitoring_loop(self) -> None:
        """バックグラウンド監視ループ"""
        while not self._stop_event.wait(self.config.evaluation_interval):
            try:
                # 定期監視処理
                asyncio.run_coroutine_threadsafe(
                    self._periodic_monitoring(), asyncio.get_event_loop()
                )

                # クリーンアップ処理
                asyncio.run_coroutine_threadsafe(
                    self._cleanup_old_data(), asyncio.get_event_loop()
                )

            except Exception as e:
                logger.error(f"Background monitoring error: {e}")

    async def _periodic_monitoring(self) -> None:
        """定期監視処理"""
        try:
            # 抑制ルールの期限切れチェック
            await self._cleanup_expired_suppressions()

            # アラートグルーピングの実行
            if self.config.enable_grouping:
                await self.group_alerts()

            # エスカレーション処理
            if self.config.enable_escalation:
                await self._process_escalations()

        except Exception as e:
            logger.error(f"Periodic monitoring failed: {e}")

    async def add_alert_rule(self, rule: AlertRule) -> None:
        """アラートルール追加"""
        self._alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.id}")

    async def remove_alert_rule(self, rule_id: str) -> None:
        """アラートルール削除"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    async def add_notification_channel(self, channel: NotificationChannel) -> None:
        """通知チャネル追加"""
        self._notification_channels[channel.id] = channel
        logger.info(f"Added notification channel: {channel.id}")

    async def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """エスカレーションポリシー追加"""
        self._escalation_policies[policy.id] = policy
        logger.info(f"Added escalation policy: {policy.id}")

    async def add_suppress_rule(self, rule: SuppressRule) -> None:
        """抑制ルール追加"""
        self._suppress_rules[rule.id] = rule
        logger.info(f"Added suppress rule: {rule.id}")

    async def evaluate_metric(self, metric_name: str, value: float) -> list[AlertTriggerEvent]:
        """メトリクス評価・アラート判定"""
        triggered_alerts: list[AlertTriggerEvent] = []

        # 抑制チェック
        if self._is_suppressed(metric_name, value):
            return triggered_alerts

        # マッチするアラートルールを検索
        for rule in self._alert_rules.values():
            if not rule.matches_metric(metric_name):
                continue

            # 条件評価
            if rule.condition.evaluate(value):
                # クールダウンチェック
                if self._is_in_cooldown(rule.id):
                    continue

                # アラート発動
                alert = self._create_alert_event(rule, metric_name, value)
                triggered_alerts.append(alert)

                # アラート記録
                await self._record_alert(alert)

                # 通知送信
                if self.config.enable_alerting:
                    await self.send_alert_notifications(alert)

        return triggered_alerts

    def _is_suppressed(self, metric: str, value: float) -> bool:
        """抑制チェック"""
        if not self.config.enable_suppression:
            return False

        # 抑制ルールの活性化チェック
        for suppress_rule in self._suppress_rules.values():
            # 抑制条件をチェックして、条件に合致する場合は抑制を活性化
            if suppress_rule.matches(metric, value):
                # 抑制ルールが新しく活性化された場合、created_atを更新
                if suppress_rule.id not in self._active_suppressions:
                    suppress_rule.created_at = datetime.now()
                    self._active_suppressions.add(suppress_rule.id)
                return True

        # 既存の抑制状態をチェック
        active_suppress_ids = set(self._active_suppressions)
        for suppress_id in active_suppress_ids:
            rule = self._suppress_rules.get(suppress_id)
            if rule is not None and not rule.is_expired():
                # 抑制が有効な期間中は全てのメトリクスを抑制
                return True
            elif rule is not None and rule.is_expired():
                # 期限切れの抑制を削除
                self._active_suppressions.discard(suppress_id)

        return False

    def _is_in_cooldown(self, rule_id: str) -> bool:
        """クールダウンチェック"""
        if rule_id not in self._last_alert_times:
            return False

        last_time = self._last_alert_times[rule_id]
        rule = self._alert_rules.get(rule_id)
        if not rule:
            return False

        cooldown_end = last_time + timedelta(seconds=rule.cooldown)
        return datetime.now() < cooldown_end

    def _create_alert_event(
        self, rule: AlertRule, metric_name: str, value: float
    ) -> AlertTriggerEvent:
        """アラートイベント作成"""
        return AlertTriggerEvent(
            rule_id=rule.id,
            rule_name=rule.name,
            metric=metric_name,
            value=value,
            threshold=rule.condition.threshold,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            triggered_at=datetime.now(),
            description=f"{rule.description}: {metric_name}={value} {rule.condition.operator.value} {rule.condition.threshold}",
            tags=rule.tags,
        )

    async def _record_alert(self, alert: AlertTriggerEvent) -> None:
        """アラート記録"""
        # アクティブアラートに追加（複数アラートの蓄積）
        self._active_alerts[alert.rule_id].append(alert)

        # 履歴に追加
        self._alert_history.append(alert)

        # 最終アラート時刻を更新
        self._last_alert_times[alert.rule_id] = alert.triggered_at

        # コールバック実行
        if self._alert_callback:
            try:
                if asyncio.iscoroutinefunction(self._alert_callback):
                    await self._alert_callback(alert.to_dict())
                else:
                    self._alert_callback(alert.to_dict())
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def send_alert_notifications(self, alert: AlertTriggerEvent) -> list[dict[str, Any]]:
        """アラート通知送信"""
        sent_notifications = []

        for channel in self._notification_channels.values():
            if not channel.can_send(alert.severity):
                continue

            try:
                notification_result = await self._send_notification(channel, alert)
                sent_notifications.append(notification_result)

                # 通知履歴に記録
                self._notification_history.append({
                    "channel_id": channel.id,
                    "alert_id": alert.rule_id,
                    "severity": alert.severity.value,
                    "sent_at": datetime.now(),
                    "success": notification_result.get("success", False),
                })

            except Exception as e:
                logger.error(f"Failed to send notification via {channel.id}: {e}")
                sent_notifications.append({
                    "channel_id": channel.id,
                    "success": False,
                    "error": str(e),
                })

        return sent_notifications

    async def _send_notification(
        self, channel: NotificationChannel, alert: AlertTriggerEvent
    ) -> dict[str, Any]:
        """個別チャネルへの通知送信"""
        if channel.type == AlertChannel.EMAIL:
            return await self._send_email_notification(channel, alert)
        elif channel.type == AlertChannel.SLACK:
            return await self._send_slack_notification(channel, alert)
        elif channel.type == AlertChannel.WEBHOOK:
            return await self._send_webhook_notification(channel, alert)
        else:
            return {"success": False, "error": f"Unsupported channel type: {channel.type}"}

    async def _send_email_notification(
        self, channel: NotificationChannel, alert: AlertTriggerEvent
    ) -> dict[str, Any]:
        """Email通知送信"""
        try:
            config = channel.config

            # メール作成
            msg = MIMEMultipart()
            msg["From"] = config["username"]
            msg["To"] = ", ".join(config["recipients"])
            msg["Subject"] = f"[{alert.severity.value}] {alert.rule_name}"

            body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value}
Metric: {alert.metric}
Value: {alert.value}
Threshold: {alert.threshold}
Description: {alert.description}
Triggered At: {alert.triggered_at}
"""
            msg.attach(MIMEText(body, "plain"))

            # SMTP送信（モック環境では実際に送信しない）
            if config.get("mock", True):
                logger.info(f"Mock email sent to {config['recipients']}")
                return {"success": True, "channel_id": channel.id}

            # 実際のSMTP送信
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()

            return {"success": True, "channel_id": channel.id}

        except Exception as e:
            return {"success": False, "channel_id": channel.id, "error": str(e)}

    async def _send_slack_notification(
        self, channel: NotificationChannel, alert: AlertTriggerEvent
    ) -> dict[str, Any]:
        """Slack通知送信"""
        try:
            config = channel.config
            webhook_url = config["webhook_url"]

            # Slackメッセージ作成
            payload = {
                "channel": config.get("channel", "#alerts"),
                "username": config.get("username", "AlertBot"),
                "icon_emoji": ":warning:" if alert.severity == AlertSeverity.WARNING else ":fire:",
                "attachments": [
                    {
                        "color": "danger" if alert.severity == AlertSeverity.CRITICAL else "warning",
                        "title": f"[{alert.severity.value}] {alert.rule_name}",
                        "text": alert.description,
                        "fields": [
                            {"title": "Metric", "value": alert.metric, "short": True},
                            {"title": "Value", "value": str(alert.value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Time", "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        ],
                    }
                ],
            }

            # Webhook送信（モック環境では実際に送信しない）
            if config.get("mock", True):
                logger.info(f"Mock Slack notification sent to {config.get('channel', '#alerts')}")
                return {"success": True, "channel_id": channel.id}

            # 実際のWebhook送信
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        return {"success": True, "channel_id": channel.id}
                    else:
                        return {"success": False, "channel_id": channel.id, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "channel_id": channel.id, "error": str(e)}

    async def _send_webhook_notification(
        self, channel: NotificationChannel, alert: AlertTriggerEvent
    ) -> dict[str, Any]:
        """Webhook通知送信"""
        try:
            config = channel.config
            url = config["url"]
            method = config.get("method", "POST")
            headers = config.get("headers", {})
            timeout = config.get("timeout", 30)

            # ペイロード作成
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
            }

            # Webhook送信（モック環境では実際に送信しない）
            if config.get("mock", True):
                logger.info(f"Mock webhook notification sent to {url}")
                return {"success": True, "channel_id": channel.id}

            # 実際のWebhook送信
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if 200 <= response.status < 300:
                        return {"success": True, "channel_id": channel.id}
                    else:
                        return {"success": False, "channel_id": channel.id, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "channel_id": channel.id, "error": str(e)}

    async def group_alerts(self) -> list[dict[str, Any]]:
        """アラートグルーピング"""
        if not self.config.enable_grouping:
            return []

        groups = defaultdict(list)

        # アクティブアラートをタグでグルーピング
        for alert_list in self._active_alerts.values():
            for alert in alert_list:
                # グルーピングキーの生成（タグベース）
                group_key = self._generate_group_key(alert)
                groups[group_key].append(alert)

        # グルーピング結果
        grouped_alerts = []
        for group_key, alerts in groups.items():
            if len(alerts) >= 1:  # 1個以上のアラートがある場合グルーピング情報を作成
                grouped_alerts.append({
                    "group_key": group_key,
                    "alert_count": len(alerts),
                    "alerts": [alert.to_dict() for alert in alerts],
                    "tags": alerts[0].tags if alerts else {},
                    "grouped_at": datetime.now(),
                })

        return grouped_alerts

    def _generate_group_key(self, alert: AlertTriggerEvent) -> str:
        """グルーピングキー生成"""
        # タグベースのグルーピングキー
        tag_keys = sorted(alert.tags.keys())
        tag_values = [f"{key}={alert.tags[key]}" for key in tag_keys]
        return "|".join(tag_values) if tag_values else f"rule_{alert.rule_id}"

    async def trigger_escalation(self, alert: AlertTriggerEvent) -> None:
        """エスカレーション開始"""
        if not self.config.enable_escalation:
            return

        # 該当するエスカレーションポリシーを検索
        for policy in self._escalation_policies.values():
            steps = policy.get_escalation_steps(alert.severity)
            if not steps:
                continue

            # エスカレーション実行
            for step in steps:
                delay = step.get("delay", 0)
                channels = step.get("channels", [])

                if delay > 0:
                    # 遅延実行
                    asyncio.create_task(
                        self._delayed_escalation(alert, channels, delay)
                    )
                else:
                    # 即座に実行
                    await self._execute_escalation_step(alert, channels)

                # エスカレーション履歴に記録
                self._escalation_history[alert.rule_id].append({
                    "step": step,
                    "executed_at": datetime.now(),
                    "alert_id": alert.rule_id,
                })

    async def _delayed_escalation(
        self, alert: AlertTriggerEvent, channels: list[str], delay: int
    ) -> None:
        """遅延エスカレーション実行"""
        await asyncio.sleep(delay)
        await self._execute_escalation_step(alert, channels)

    async def _execute_escalation_step(
        self, alert: AlertTriggerEvent, channels: list[str]
    ) -> None:
        """エスカレーション手順実行"""
        for channel_id in channels:
            channel = self._notification_channels.get(channel_id)
            if channel and channel.enabled:
                await self._send_notification(channel, alert)

    async def get_escalation_history(self, rule_id: str) -> list[dict[str, Any]]:
        """エスカレーション履歴取得"""
        return self._escalation_history.get(rule_id, [])

    async def get_alert_history(
        self, start_time: datetime, end_time: datetime
    ) -> list[AlertTriggerEvent]:
        """アラート履歴取得"""
        return [
            alert for alert in self._alert_history
            if start_time <= alert.triggered_at <= end_time
        ]

    async def _cleanup_expired_suppressions(self) -> None:
        """期限切れ抑制ルールのクリーンアップ"""
        expired_rules = [
            rule_id for rule_id, rule in self._suppress_rules.items()
            if rule.is_expired()
        ]

        for rule_id in expired_rules:
            del self._suppress_rules[rule_id]
            logger.info(f"Removed expired suppress rule: {rule_id}")

    async def _process_escalations(self) -> None:
        """エスカレーション処理"""
        # 長時間継続しているアラートに対してエスカレーション実行
        cutoff_time = datetime.now() - timedelta(minutes=15)

        for alert_list in self._active_alerts.values():
            for alert in alert_list:
                if alert.triggered_at <= cutoff_time and alert.severity == AlertSeverity.CRITICAL:
                    # 15分以上継続しているクリティカルアラートはエスカレーション
                    await self.trigger_escalation(alert)

    async def _cleanup_old_data(self) -> None:
        """古いデータのクリーンアップ"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config.alert_retention_days)

            # 古いアラート履歴をクリーンアップ
            old_alerts = [
                alert for alert in self._alert_history
                if alert.triggered_at < cutoff_time
            ]

            for alert in old_alerts:
                self._alert_history.remove(alert)

            # 古いエスカレーション履歴をクリーンアップ
            for rule_id in list(self._escalation_history.keys()):
                self._escalation_history[rule_id] = [
                    entry for entry in self._escalation_history[rule_id]
                    if entry["executed_at"] >= cutoff_time
                ]

                if not self._escalation_history[rule_id]:
                    del self._escalation_history[rule_id]

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

    def set_alert_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """アラートコールバック設定"""
        self._alert_callback = callback

    def get_monitoring_status(self) -> dict[str, Any]:
        """監視状況取得"""
        return {
            "is_monitoring": self._is_monitoring,
            "alert_rules_count": len(self._alert_rules),
            "notification_channels_count": len(self._notification_channels),
            "active_alerts_count": len(self._active_alerts),
            "alert_history_count": len(self._alert_history),
            "escalation_policies_count": len(self._escalation_policies),
            "suppress_rules_count": len(self._suppress_rules),
            "config": asdict(self.config),
        }
