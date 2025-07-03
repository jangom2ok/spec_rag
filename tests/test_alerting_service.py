"""アラート機能サービステスト

TDD実装：パフォーマンス・エラー監視アラート機能
- しきい値ベースアラート: レスポンス時間、エラー率、CPU/メモリ使用率
- 複合条件アラート: 複数メトリクスの組み合わせ
- アラートエスカレーション: 重要度別の通知先設定
- 抑制・統合機能: 類似アラートの重複抑制
- 通知チャネル: Email、Slack、Webhook統合
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from app.services.alerting_service import (
    AlertChannel,
    AlertCondition,
    AlertingConfig,
    AlertingService,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertTriggerEvent,
    ComparisonOperator,
    EscalationPolicy,
    NotificationChannel,
    SuppressRule,
)


@pytest.fixture
def basic_alerting_config() -> AlertingConfig:
    """基本的なアラート設定"""
    return AlertingConfig(
        enable_alerting=True,
        evaluation_interval=60,  # 60秒間隔
        default_severity=AlertSeverity.WARNING,
        max_alerts_per_minute=10,
        alert_retention_days=30,
    )


@pytest.fixture
def comprehensive_alerting_config() -> AlertingConfig:
    """包括的なアラート設定"""
    return AlertingConfig(
        enable_alerting=True,
        evaluation_interval=30,  # 30秒間隔
        default_severity=AlertSeverity.WARNING,
        max_alerts_per_minute=20,
        alert_retention_days=90,
        enable_escalation=True,
        enable_suppression=True,
        enable_grouping=True,
        grouping_window=300,  # 5分間のグルーピング
        notification_cooldown=600,  # 10分間のクールダウン
    )


@pytest.fixture
def sample_alert_rules() -> list[AlertRule]:
    """サンプルアラートルール"""
    return [
        AlertRule(
            id="high_response_time",
            name="High Response Time Alert",
            description="Alert when API response time exceeds threshold",
            metric="response_time",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=5000.0,  # 5秒
                duration=300,  # 5分間継続
            ),
            severity=AlertSeverity.WARNING,
            enabled=True,
            tags={"service": "api", "category": "performance"},
        ),
        AlertRule(
            id="high_error_rate",
            name="High Error Rate Alert",
            description="Alert when error rate exceeds threshold",
            metric="error_rate",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=0.05,  # 5%
                duration=120,  # 2分間継続
            ),
            severity=AlertSeverity.CRITICAL,
            enabled=True,
            tags={"service": "api", "category": "error"},
        ),
        AlertRule(
            id="cpu_usage_alert",
            name="High CPU Usage Alert",
            description="Alert when CPU usage is consistently high",
            metric="cpu_usage",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=0.8,  # 80%
                duration=600,  # 10分間継続
            ),
            severity=AlertSeverity.WARNING,
            enabled=True,
            tags={"service": "system", "category": "resource"},
        ),
    ]


@pytest.fixture
def notification_channels() -> list[NotificationChannel]:
    """通知チャネル設定"""
    return [
        NotificationChannel(
            id="email_alerts",
            name="Email Notifications",
            type=AlertChannel.EMAIL,
            config={
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "password",
                "recipients": ["admin@example.com", "ops@example.com"],
            },
            enabled=True,
        ),
        NotificationChannel(
            id="slack_alerts",
            name="Slack Notifications",
            type=AlertChannel.SLACK,
            config={
                "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
                "channel": "#alerts",
                "username": "AlertBot",
            },
            enabled=True,
        ),
        NotificationChannel(
            id="webhook_alerts",
            name="Webhook Notifications",
            type=AlertChannel.WEBHOOK,
            config={
                "url": "https://api.example.com/alerts",
                "method": "POST",
                "headers": {"Authorization": "Bearer token123"},
                "timeout": 30,
            },
            enabled=True,
        ),
    ]


class TestAlertingConfig:
    """アラート設定のテスト"""

    @pytest.mark.unit
    def test_basic_config_creation(self):
        """基本設定の作成"""
        config = AlertingConfig(
            enable_alerting=True,
            evaluation_interval=60,
        )

        assert config.enable_alerting is True
        assert config.evaluation_interval == 60
        assert config.default_severity == AlertSeverity.WARNING
        assert not config.enable_escalation
        assert not config.enable_suppression

    @pytest.mark.unit
    def test_config_validation_success(self):
        """設定値のバリデーション（成功）"""
        config = AlertingConfig(
            enable_alerting=True,
            evaluation_interval=30,
            max_alerts_per_minute=50,
            alert_retention_days=60,
        )

        assert config.evaluation_interval == 30
        assert config.max_alerts_per_minute == 50
        assert config.alert_retention_days == 60

    @pytest.mark.unit
    def test_config_validation_invalid_interval(self):
        """無効な評価間隔のバリデーション"""
        with pytest.raises(ValueError, match="evaluation_interval must be greater than 0"):
            AlertingConfig(
                enable_alerting=True,
                evaluation_interval=0,
            )

    @pytest.mark.unit
    def test_config_validation_invalid_retention(self):
        """無効な保持期間のバリデーション"""
        with pytest.raises(ValueError, match="alert_retention_days must be greater than 0"):
            AlertingConfig(
                enable_alerting=True,
                evaluation_interval=60,
                alert_retention_days=-1,
            )


class TestAlertRule:
    """アラートルールのテスト"""

    @pytest.mark.unit
    def test_alert_rule_creation(self):
        """アラートルール作成"""
        rule = AlertRule(
            id="test_rule",
            name="Test Alert",
            description="Test alert rule",
            metric="response_time",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=1000.0,
                duration=120,
            ),
            severity=AlertSeverity.WARNING,
        )

        assert rule.id == "test_rule"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.condition.threshold == 1000.0
        assert rule.enabled is True  # デフォルト値

    @pytest.mark.unit
    def test_alert_condition_evaluation(self):
        """アラート条件評価"""
        condition = AlertCondition(
            operator=ComparisonOperator.GREATER_THAN,
            threshold=100.0,
            duration=60,
        )

        # 閾値を超える値
        assert condition.evaluate(150.0) is True
        assert condition.evaluate(100.1) is True

        # 閾値以下の値
        assert condition.evaluate(100.0) is False
        assert condition.evaluate(50.0) is False

    @pytest.mark.unit
    def test_different_comparison_operators(self):
        """異なる比較演算子のテスト"""
        # GREATER_THAN
        gt_condition = AlertCondition(
            operator=ComparisonOperator.GREATER_THAN,
            threshold=50.0,
            duration=60,
        )
        assert gt_condition.evaluate(60.0) is True
        assert gt_condition.evaluate(40.0) is False

        # LESS_THAN
        lt_condition = AlertCondition(
            operator=ComparisonOperator.LESS_THAN,
            threshold=50.0,
            duration=60,
        )
        assert lt_condition.evaluate(40.0) is True
        assert lt_condition.evaluate(60.0) is False

        # EQUAL
        eq_condition = AlertCondition(
            operator=ComparisonOperator.EQUAL,
            threshold=50.0,
            duration=60,
        )
        assert eq_condition.evaluate(50.0) is True
        assert eq_condition.evaluate(49.9) is False


class TestAlertingService:
    """アラートサービスのテスト"""

    @pytest.mark.unit
    def test_service_initialization(self, basic_alerting_config: AlertingConfig):
        """サービスの初期化"""
        service = AlertingService(config=basic_alerting_config)

        assert service.config == basic_alerting_config
        assert service._is_monitoring is False
        assert len(service._alert_rules) == 0
        assert len(service._active_alerts) == 0

    @pytest.mark.unit
    async def test_add_alert_rule(
        self, basic_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """アラートルール追加"""
        service = AlertingService(config=basic_alerting_config)

        for rule in sample_alert_rules:
            await service.add_alert_rule(rule)

        assert len(service._alert_rules) == len(sample_alert_rules)
        assert "high_response_time" in service._alert_rules
        assert "high_error_rate" in service._alert_rules

    @pytest.mark.unit
    async def test_remove_alert_rule(
        self, basic_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """アラートルール削除"""
        service = AlertingService(config=basic_alerting_config)

        # ルール追加
        for rule in sample_alert_rules:
            await service.add_alert_rule(rule)

        # 1つのルールを削除
        await service.remove_alert_rule("high_response_time")

        assert len(service._alert_rules) == len(sample_alert_rules) - 1
        assert "high_response_time" not in service._alert_rules
        assert "high_error_rate" in service._alert_rules

    @pytest.mark.unit
    async def test_start_monitoring(self, basic_alerting_config: AlertingConfig):
        """監視開始"""
        service = AlertingService(config=basic_alerting_config)

        await service.start_monitoring()

        assert service._is_monitoring is True

    @pytest.mark.unit
    async def test_stop_monitoring(self, basic_alerting_config: AlertingConfig):
        """監視停止"""
        service = AlertingService(config=basic_alerting_config)

        await service.start_monitoring()
        await service.stop_monitoring()

        assert service._is_monitoring is False


class TestMetricEvaluation:
    """メトリクス評価のテスト"""

    @pytest.mark.unit
    async def test_evaluate_single_metric(
        self, basic_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """単一メトリクス評価"""
        service = AlertingService(config=basic_alerting_config)

        # レスポンス時間ルールを追加
        response_time_rule = sample_alert_rules[0]  # high_response_time
        await service.add_alert_rule(response_time_rule)

        # 閾値を超えるメトリクス
        high_response_time = 6000.0  # 6秒（閾値5秒を超過）
        triggered_alerts = await service.evaluate_metric("response_time", high_response_time)

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].rule_id == "high_response_time"
        assert triggered_alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.unit
    async def test_evaluate_multiple_metrics(
        self, basic_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """複数メトリクス評価"""
        service = AlertingService(config=basic_alerting_config)

        # 全ルールを追加
        for rule in sample_alert_rules:
            await service.add_alert_rule(rule)

        # 複数メトリクスを同時評価
        metrics = {
            "response_time": 6000.0,  # 閾値超過
            "error_rate": 0.08,       # 閾値超過
            "cpu_usage": 0.60,        # 正常範囲
        }

        all_triggered_alerts = []
        for metric_name, value in metrics.items():
            alerts = await service.evaluate_metric(metric_name, value)
            all_triggered_alerts.extend(alerts)

        # 2つのアラートが発動
        assert len(all_triggered_alerts) == 2
        rule_ids = [alert.rule_id for alert in all_triggered_alerts]
        assert "high_response_time" in rule_ids
        assert "high_error_rate" in rule_ids

    @pytest.mark.unit
    async def test_metric_within_threshold(
        self, basic_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """閾値内メトリクス（アラート発動なし）"""
        service = AlertingService(config=basic_alerting_config)

        response_time_rule = sample_alert_rules[0]
        await service.add_alert_rule(response_time_rule)

        # 閾値以下のメトリクス
        normal_response_time = 2000.0  # 2秒（閾値5秒以下）
        triggered_alerts = await service.evaluate_metric("response_time", normal_response_time)

        assert len(triggered_alerts) == 0


class TestNotificationSystem:
    """通知システムのテスト"""

    @pytest.mark.unit
    async def test_add_notification_channel(
        self, basic_alerting_config: AlertingConfig, notification_channels: list[NotificationChannel]
    ):
        """通知チャネル追加"""
        service = AlertingService(config=basic_alerting_config)

        for channel in notification_channels:
            await service.add_notification_channel(channel)

        assert len(service._notification_channels) == len(notification_channels)
        assert "email_alerts" in service._notification_channels
        assert "slack_alerts" in service._notification_channels

    @pytest.mark.unit
    async def test_send_alert_notification(
        self, basic_alerting_config: AlertingConfig, notification_channels: list[NotificationChannel]
    ):
        """アラート通知送信"""
        service = AlertingService(config=basic_alerting_config)

        # 通知チャネル追加
        for channel in notification_channels:
            await service.add_notification_channel(channel)

        # テストアラート作成
        alert = AlertTriggerEvent(
            rule_id="test_rule",
            rule_name="Test Alert",
            metric="response_time",
            value=6000.0,
            threshold=5000.0,
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            triggered_at=datetime.now(),
            description="Response time exceeded threshold",
        )

        # 通知送信（モック環境）
        sent_notifications = await service.send_alert_notifications(alert)

        # 有効な通知チャネル数と一致
        enabled_channels = [ch for ch in notification_channels if ch.enabled]
        assert len(sent_notifications) == len(enabled_channels)

    @pytest.mark.unit
    async def test_notification_channel_filtering(
        self, basic_alerting_config: AlertingConfig, notification_channels: list[NotificationChannel]
    ):
        """通知チャネルフィルタリング"""
        service = AlertingService(config=basic_alerting_config)

        # 1つのチャネルを無効化
        notification_channels[0].enabled = False

        for channel in notification_channels:
            await service.add_notification_channel(channel)

        alert = AlertTriggerEvent(
            rule_id="test_rule",
            rule_name="Test Alert",
            metric="response_time",
            value=6000.0,
            threshold=5000.0,
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            triggered_at=datetime.now(),
            description="Test alert",
        )

        sent_notifications = await service.send_alert_notifications(alert)

        # 有効なチャネルのみ通知送信
        enabled_count = sum(1 for ch in notification_channels if ch.enabled)
        assert len(sent_notifications) == enabled_count


class TestAlertSuppression:
    """アラート抑制のテスト"""

    @pytest.mark.unit
    async def test_basic_alert_suppression(self, comprehensive_alerting_config: AlertingConfig):
        """基本的なアラート抑制"""
        service = AlertingService(config=comprehensive_alerting_config)

        # 抑制ルール追加
        suppress_rule = SuppressRule(
            id="suppress_response_time",
            name="Suppress Response Time Alerts",
            conditions=[
                {"metric": "response_time", "operator": "gt", "value": 10000}
            ],
            duration=1800,  # 30分間抑制
            enabled=True,
        )
        await service.add_suppress_rule(suppress_rule)

        # アラートルール追加
        alert_rule = AlertRule(
            id="response_time_alert",
            name="Response Time Alert",
            description="Response time alert",
            metric="response_time",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=5000.0,
                duration=60,
            ),
            severity=AlertSeverity.WARNING,
        )
        await service.add_alert_rule(alert_rule)

        # 抑制条件に該当するメトリクス
        suppressed_value = 12000.0  # 12秒
        triggered_alerts = await service.evaluate_metric("response_time", suppressed_value)

        # アラートは抑制される
        assert len(triggered_alerts) == 0

    @pytest.mark.unit
    async def test_alert_suppression_expiry(self, comprehensive_alerting_config: AlertingConfig):
        """アラート抑制の期限切れ"""
        comprehensive_alerting_config.evaluation_interval = 1  # 1秒間隔でテスト
        service = AlertingService(config=comprehensive_alerting_config)

        # 短期間の抑制ルール
        suppress_rule = SuppressRule(
            id="short_suppress",
            name="Short Suppression",
            conditions=[
                {"metric": "error_rate", "operator": "gt", "value": 0.1}
            ],
            duration=2,  # 2秒間抑制
            enabled=True,
        )
        await service.add_suppress_rule(suppress_rule)

        alert_rule = AlertRule(
            id="error_rate_alert",
            name="Error Rate Alert",
            description="Error rate alert",
            metric="error_rate",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=0.05,
                duration=1,
            ),
            severity=AlertSeverity.CRITICAL,
        )
        await service.add_alert_rule(alert_rule)

        # 抑制トリガー
        await service.evaluate_metric("error_rate", 0.15)

        # 抑制期間中
        alerts_during_suppression = await service.evaluate_metric("error_rate", 0.08)
        assert len(alerts_during_suppression) == 0

        # 抑制期間経過を待機
        await asyncio.sleep(3)

        # 抑制期間後
        alerts_after_expiry = await service.evaluate_metric("error_rate", 0.08)
        assert len(alerts_after_expiry) == 1


class TestEscalationPolicy:
    """エスカレーションポリシーのテスト"""

    @pytest.mark.unit
    async def test_escalation_policy_creation(self):
        """エスカレーションポリシー作成"""
        policy = EscalationPolicy(
            id="critical_escalation",
            name="Critical Alert Escalation",
            rules=[
                {
                    "severity": "CRITICAL",
                    "delay": 0,
                    "channels": ["email_alerts"],
                },
                {
                    "severity": "CRITICAL",
                    "delay": 300,  # 5分後
                    "channels": ["slack_alerts"],
                },
                {
                    "severity": "CRITICAL",
                    "delay": 900,  # 15分後
                    "channels": ["webhook_alerts"],
                },
            ],
            enabled=True,
        )

        assert policy.id == "critical_escalation"
        assert len(policy.rules) == 3
        assert policy.enabled is True

    @pytest.mark.unit
    async def test_escalation_execution(
        self, comprehensive_alerting_config: AlertingConfig, notification_channels: list[NotificationChannel]
    ):
        """エスカレーション実行"""
        service = AlertingService(config=comprehensive_alerting_config)

        # 通知チャネル追加
        for channel in notification_channels:
            await service.add_notification_channel(channel)

        # エスカレーションポリシー追加
        policy = EscalationPolicy(
            id="test_escalation",
            name="Test Escalation",
            rules=[
                {
                    "severity": "CRITICAL",
                    "delay": 0,
                    "channels": ["email_alerts"],
                },
                {
                    "severity": "CRITICAL",
                    "delay": 1,  # 1秒後（テスト用）
                    "channels": ["slack_alerts"],
                },
            ],
            enabled=True,
        )
        await service.add_escalation_policy(policy)

        # クリティカルアラート作成
        alert = AlertTriggerEvent(
            rule_id="critical_test",
            rule_name="Critical Test Alert",
            metric="error_rate",
            value=0.95,
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            triggered_at=datetime.now(),
            description="Critical error rate",
        )

        # エスカレーション開始
        await service.trigger_escalation(alert)

        # 短時間待機してエスカレーション実行
        await asyncio.sleep(2)

        # エスカレーション履歴確認
        escalation_history = await service.get_escalation_history(alert.rule_id)
        assert len(escalation_history) >= 1


class TestAlertGrouping:
    """アラートグルーピングのテスト"""

    @pytest.mark.unit
    async def test_alert_grouping_by_tags(self, comprehensive_alerting_config: AlertingConfig):
        """タグによるアラートグルーピング"""
        service = AlertingService(config=comprehensive_alerting_config)

        # 同じタグを持つ複数のアラートルール
        rules = [
            AlertRule(
                id="api_response_time",
                name="API Response Time",
                description="API response time alert",
                metric="api_response_time",
                condition=AlertCondition(
                    operator=ComparisonOperator.GREATER_THAN,
                    threshold=2000.0,
                    duration=60,
                ),
                severity=AlertSeverity.WARNING,
                tags={"service": "api", "environment": "production"},
            ),
            AlertRule(
                id="api_error_rate",
                name="API Error Rate",
                description="API error rate alert",
                metric="api_error_rate",
                condition=AlertCondition(
                    operator=ComparisonOperator.GREATER_THAN,
                    threshold=0.02,
                    duration=120,
                ),
                severity=AlertSeverity.WARNING,
                tags={"service": "api", "environment": "production"},
            ),
        ]

        for rule in rules:
            await service.add_alert_rule(rule)

        # 両方のメトリクスでアラート発動
        await service.evaluate_metric("api_response_time", 3000.0)
        await service.evaluate_metric("api_error_rate", 0.05)

        # グルーピング実行
        groups = await service.group_alerts()

        # 同じタグでグルーピングされることを確認
        api_groups = [g for g in groups if g.get("tags", {}).get("service") == "api"]
        assert len(api_groups) >= 1

    @pytest.mark.unit
    async def test_alert_grouping_window(self, comprehensive_alerting_config: AlertingConfig):
        """グルーピング時間窓のテスト"""
        comprehensive_alerting_config.grouping_window = 2  # 2秒のグルーピング窓
        service = AlertingService(config=comprehensive_alerting_config)

        alert_rule = AlertRule(
            id="test_grouping",
            name="Test Grouping",
            description="Test grouping alert",
            metric="test_metric",
            condition=AlertCondition(
                operator=ComparisonOperator.GREATER_THAN,
                threshold=50.0,
                duration=1,
            ),
            severity=AlertSeverity.WARNING,
            tags={"group": "test"},
            cooldown=0,  # クールダウンを無効化
        )
        await service.add_alert_rule(alert_rule)

        # 短時間内に複数回アラート発動
        await service.evaluate_metric("test_metric", 60.0)
        await asyncio.sleep(0.5)
        await service.evaluate_metric("test_metric", 70.0)
        await asyncio.sleep(0.5)
        await service.evaluate_metric("test_metric", 80.0)

        # グルーピング実行
        groups = await service.group_alerts()

        # 同じグループにまとめられることを確認
        test_groups = [g for g in groups if g.get("tags", {}).get("group") == "test"]
        assert len(test_groups) == 1
        assert test_groups[0]["alert_count"] >= 3


class TestAlertingIntegration:
    """アラート統合テスト"""

    @pytest.mark.integration
    async def test_end_to_end_alerting_flow(
        self,
        comprehensive_alerting_config: AlertingConfig,
        sample_alert_rules: list[AlertRule],
        notification_channels: list[NotificationChannel],
    ):
        """エンドツーエンドアラートフロー"""
        service = AlertingService(config=comprehensive_alerting_config)

        # セットアップ
        for rule in sample_alert_rules:
            await service.add_alert_rule(rule)

        for channel in notification_channels:
            await service.add_notification_channel(channel)

        await service.start_monitoring()

        try:
            # アラート発動シナリオ
            triggered_alerts = []

            # 1. レスポンス時間アラート
            response_alerts = await service.evaluate_metric("response_time", 6000.0)
            triggered_alerts.extend(response_alerts)

            # 2. エラー率アラート
            error_alerts = await service.evaluate_metric("error_rate", 0.08)
            triggered_alerts.extend(error_alerts)

            # 3. CPU使用率（正常範囲）
            cpu_alerts = await service.evaluate_metric("cpu_usage", 0.70)
            triggered_alerts.extend(cpu_alerts)

            assert len(triggered_alerts) == 2  # レスポンス時間とエラー率

            # 通知送信
            for alert in triggered_alerts:
                notifications = await service.send_alert_notifications(alert)
                assert len(notifications) > 0

            # アラート履歴確認
            alert_history = await service.get_alert_history(
                start_time=datetime.now() - timedelta(minutes=5),
                end_time=datetime.now(),
            )
            assert len(alert_history) >= 2

        finally:
            await service.stop_monitoring()

    @pytest.mark.integration
    async def test_real_time_monitoring_system(
        self, comprehensive_alerting_config: AlertingConfig, sample_alert_rules: list[AlertRule]
    ):
        """リアルタイム監視システム"""
        service = AlertingService(config=comprehensive_alerting_config)

        for rule in sample_alert_rules:
            await service.add_alert_rule(rule)

        triggered_alerts = []

        # アラートコールバック
        async def alert_callback(alert_data: dict[str, Any]):
            triggered_alerts.append(alert_data)

        service.set_alert_callback(alert_callback)
        await service.start_monitoring()

        try:
            # メトリクスストリーム
            test_metrics = [
                ("response_time", 7000.0),  # アラート発動
                ("response_time", 3000.0),  # 正常
                ("error_rate", 0.10),       # アラート発動
                ("cpu_usage", 0.85),        # アラート発動
            ]

            for metric_name, value in test_metrics:
                await service.evaluate_metric(metric_name, value)
                await asyncio.sleep(0.1)

            # 短時間待機
            await asyncio.sleep(0.5)

            # 複数のアラートが発動
            assert len(triggered_alerts) >= 3

            # アラートの内容確認
            alert_rules = [alert["rule_id"] for alert in triggered_alerts]
            assert "high_response_time" in alert_rules
            assert "high_error_rate" in alert_rules
            assert "cpu_usage_alert" in alert_rules

        finally:
            await service.stop_monitoring()
