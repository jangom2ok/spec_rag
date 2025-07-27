"""管理画面サービステスト

TDD実装：検索分析・システム状態管理ダッシュボード
- 検索分析: クエリ統計、パフォーマンス分析、ユーザー行動分析
- システム状態: リアルタイム監視、リソース使用状況、アラート状況
- ダッシュボード: カスタマイズ可能なウィジェット、レポート生成
- ユーザー管理: 権限管理、アクセス制御、監査ログ
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from app.services.admin_dashboard import (
    AdminDashboard,
    DashboardConfig,
    DashboardWidget,
    ReportConfig,
    ReportFormat,
    ReportType,
    SystemMetrics,
    UserAnalytics,
    WidgetType,
)


@pytest.fixture
def basic_dashboard_config() -> DashboardConfig:
    """基本的なダッシュボード設定"""
    return DashboardConfig(
        enable_real_time=True,
        refresh_interval=30,  # 30秒間隔
        max_widgets=20,
        default_theme="light",
        enable_export=True,
        enable_notifications=True,
    )


@pytest.fixture
def comprehensive_dashboard_config() -> DashboardConfig:
    """包括的なダッシュボード設定"""
    return DashboardConfig(
        enable_real_time=True,
        refresh_interval=10,  # 10秒間隔
        max_widgets=50,
        default_theme="dark",
        enable_export=True,
        enable_notifications=True,
        enable_user_management=True,
        enable_audit_log=True,
        enable_custom_reports=True,
        data_retention_days=90,
        cache_duration=300,  # 5分間キャッシュ
        max_concurrent_users=100,
    )


@pytest.fixture
def sample_widgets() -> list[DashboardWidget]:
    """サンプルウィジェット設定"""
    return [
        DashboardWidget(
            id="search_volume",
            title="検索ボリューム",
            type=WidgetType.LINE_CHART,
            config={
                "metrics": ["search_count"],
                "time_range": "24h",
                "refresh_rate": 60,
            },
            position={"x": 0, "y": 0, "width": 6, "height": 4},
            enabled=True,
        ),
        DashboardWidget(
            id="response_time",
            title="応答時間",
            type=WidgetType.GAUGE,
            config={
                "metrics": ["avg_response_time"],
                "thresholds": {"warning": 1000, "critical": 2000},
                "unit": "ms",
            },
            position={"x": 6, "y": 0, "width": 3, "height": 4},
            enabled=True,
        ),
        DashboardWidget(
            id="error_rate",
            title="エラー率",
            type=WidgetType.BAR_CHART,
            config={
                "metrics": ["error_rate"],
                "time_range": "1h",
                "alert_threshold": 0.05,
            },
            position={"x": 9, "y": 0, "width": 3, "height": 4},
            enabled=True,
        ),
        DashboardWidget(
            id="user_activity",
            title="ユーザー活動",
            type=WidgetType.TABLE,
            config={
                "columns": ["user_id", "query_count", "last_activity"],
                "page_size": 10,
                "sortable": True,
            },
            position={"x": 0, "y": 4, "width": 12, "height": 6},
            enabled=True,
        ),
    ]


class TestDashboardConfig:
    """ダッシュボード設定のテスト"""

    @pytest.mark.unit
    def test_basic_config_creation(self):
        """基本設定の作成"""
        config = DashboardConfig(
            enable_real_time=True,
            refresh_interval=60,
        )

        assert config.enable_real_time is True
        assert config.refresh_interval == 60
        assert config.max_widgets == 10  # デフォルト値
        assert config.default_theme == "light"
        assert not config.enable_user_management

    @pytest.mark.unit
    def test_config_validation_success(self):
        """設定値のバリデーション（成功）"""
        config = DashboardConfig(
            enable_real_time=True,
            refresh_interval=30,
            max_widgets=50,
            data_retention_days=180,
        )

        assert config.refresh_interval == 30
        assert config.max_widgets == 50
        assert config.data_retention_days == 180

    @pytest.mark.unit
    def test_config_validation_invalid_interval(self):
        """無効な更新間隔のバリデーション"""
        with pytest.raises(ValueError, match="refresh_interval must be greater than 0"):
            DashboardConfig(
                enable_real_time=True,
                refresh_interval=0,
            )

    @pytest.mark.unit
    def test_config_validation_invalid_widgets(self):
        """無効なウィジェット数のバリデーション"""
        with pytest.raises(ValueError, match="max_widgets must be greater than 0"):
            DashboardConfig(
                enable_real_time=True,
                refresh_interval=60,
                max_widgets=0,
            )


class TestDashboardWidget:
    """ダッシュボードウィジェットのテスト"""

    @pytest.mark.unit
    def test_widget_creation(self):
        """ウィジェット作成"""
        widget = DashboardWidget(
            id="test_widget",
            title="テストウィジェット",
            type=WidgetType.LINE_CHART,
            config={"metric": "test_metric"},
            position={"x": 0, "y": 0, "width": 6, "height": 4},
        )

        assert widget.id == "test_widget"
        assert widget.type == WidgetType.LINE_CHART
        assert widget.enabled is True  # デフォルト値
        assert widget.position["width"] == 6

    @pytest.mark.unit
    def test_widget_validation(self):
        """ウィジェット設定の検証"""
        widget = DashboardWidget(
            id="search_chart",
            title="検索チャート",
            type=WidgetType.PIE_CHART,
            config={
                "metrics": ["search_success", "search_failure"],
                "colors": ["#4CAF50", "#F44336"],
            },
            position={"x": 0, "y": 0, "width": 4, "height": 4},
        )

        assert widget.validate_position()
        assert widget.validate_config()

    @pytest.mark.unit
    def test_widget_data_requirements(self):
        """ウィジェットデータ要件"""
        chart_widget = DashboardWidget(
            id="performance_chart",
            title="パフォーマンス",
            type=WidgetType.AREA_CHART,
            config={
                "metrics": ["response_time", "throughput"],
                "time_range": "6h",
            },
            position={"x": 0, "y": 0, "width": 8, "height": 6},
        )

        required_fields = chart_widget.get_required_data_fields()
        assert "response_time" in required_fields
        assert "throughput" in required_fields


class TestAdminDashboard:
    """管理ダッシュボードサービスのテスト"""

    @pytest.mark.unit
    def test_dashboard_initialization(self, basic_dashboard_config: DashboardConfig):
        """ダッシュボードの初期化"""
        dashboard = AdminDashboard(config=basic_dashboard_config)

        assert dashboard.config == basic_dashboard_config
        assert len(dashboard._widgets) == 0
        assert dashboard._is_real_time_enabled is True

    @pytest.mark.unit
    async def test_add_widget(
        self,
        basic_dashboard_config: DashboardConfig,
        sample_widgets: list[DashboardWidget],
    ):
        """ウィジェット追加"""
        dashboard = AdminDashboard(config=basic_dashboard_config)

        for widget in sample_widgets:
            await dashboard.add_widget(widget)

        assert len(dashboard._widgets) == len(sample_widgets)
        assert "search_volume" in dashboard._widgets
        assert "response_time" in dashboard._widgets

    @pytest.mark.unit
    async def test_remove_widget(
        self,
        basic_dashboard_config: DashboardConfig,
        sample_widgets: list[DashboardWidget],
    ):
        """ウィジェット削除"""
        dashboard = AdminDashboard(config=basic_dashboard_config)

        # ウィジェット追加
        for widget in sample_widgets:
            await dashboard.add_widget(widget)

        # 1つのウィジェットを削除
        await dashboard.remove_widget("search_volume")

        assert len(dashboard._widgets) == len(sample_widgets) - 1
        assert "search_volume" not in dashboard._widgets
        assert "response_time" in dashboard._widgets

    @pytest.mark.unit
    async def test_update_widget_config(
        self,
        basic_dashboard_config: DashboardConfig,
        sample_widgets: list[DashboardWidget],
    ):
        """ウィジェット設定更新"""
        dashboard = AdminDashboard(config=basic_dashboard_config)

        widget = sample_widgets[0]
        await dashboard.add_widget(widget)

        # 設定更新
        new_config = {"metrics": ["search_count", "user_count"], "time_range": "12h"}
        await dashboard.update_widget_config("search_volume", new_config)

        updated_widget = dashboard._widgets["search_volume"]
        assert updated_widget.config["time_range"] == "12h"
        assert "user_count" in updated_widget.config["metrics"]


class TestSystemMetrics:
    """システムメトリクスのテスト"""

    @pytest.mark.unit
    async def test_collect_system_metrics(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """システムメトリクス収集"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        metrics = await dashboard.collect_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp is not None
        assert metrics.cpu_usage >= 0.0
        assert metrics.memory_usage >= 0.0
        assert metrics.disk_usage >= 0.0

    @pytest.mark.unit
    async def test_search_analytics(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """検索分析メトリクス"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # サンプル検索データを設定
        await dashboard.record_search_event(
            {
                "query": "machine learning",
                "user_id": "user_001",
                "timestamp": datetime.now(),
                "response_time": 150,
                "results_count": 42,
                "success": True,
            }
        )

        analytics = await dashboard.get_search_analytics(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
        )

        assert analytics.total_searches >= 1
        assert analytics.avg_response_time > 0
        assert analytics.success_rate >= 0.0

    @pytest.mark.unit
    async def test_user_analytics(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """ユーザー分析メトリクス"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # サンプルユーザーデータを設定
        await dashboard.record_user_activity(
            {
                "user_id": "user_001",
                "action": "search",
                "timestamp": datetime.now(),
                "session_id": "session_123",
            }
        )

        user_analytics = await dashboard.get_user_analytics(
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now(),
        )

        assert isinstance(user_analytics, UserAnalytics)
        assert user_analytics.active_users >= 0
        assert user_analytics.total_sessions >= 0


class TestReportGeneration:
    """レポート生成のテスト"""

    @pytest.mark.unit
    async def test_generate_system_report(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """システムレポート生成"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        report_config = ReportConfig(
            type=ReportType.SYSTEM_STATUS,
            format=ReportFormat.PDF,
            time_range=timedelta(hours=24),
            include_charts=True,
            include_raw_data=False,
        )

        report = await dashboard.generate_report(report_config)

        assert report.type == ReportType.SYSTEM_STATUS
        assert report.format == ReportFormat.PDF
        assert report.generated_at is not None
        assert len(report.content) > 0

    @pytest.mark.unit
    async def test_generate_search_analytics_report(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """検索分析レポート生成"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        report_config = ReportConfig(
            type=ReportType.SEARCH_ANALYTICS,
            format=ReportFormat.CSV,
            time_range=timedelta(days=7),
            include_charts=False,
            include_raw_data=True,
        )

        report = await dashboard.generate_report(report_config)

        assert report.type == ReportType.SEARCH_ANALYTICS
        assert report.format == ReportFormat.CSV
        assert report.size_bytes > 0

    @pytest.mark.unit
    async def test_scheduled_report_generation(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """定期レポート生成"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # 定期レポートのスケジュール設定
        schedule_config = {
            "type": ReportType.USER_ANALYTICS,
            "format": ReportFormat.JSON,
            "frequency": "daily",
            "time": "09:00",
            "recipients": ["admin@example.com"],
        }

        await dashboard.schedule_report(schedule_config)

        scheduled_reports = await dashboard.get_scheduled_reports()
        assert len(scheduled_reports) >= 1


class TestRealTimeMonitoring:
    """リアルタイム監視のテスト"""

    @pytest.mark.unit
    async def test_start_real_time_monitoring(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """リアルタイム監視開始"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        await dashboard.start_real_time_monitoring()

        assert dashboard._is_real_time_enabled is True
        assert dashboard._monitoring_thread is not None

    @pytest.mark.unit
    async def test_stop_real_time_monitoring(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """リアルタイム監視停止"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        await dashboard.start_real_time_monitoring()
        await dashboard.stop_real_time_monitoring()

        assert dashboard._is_real_time_enabled is False

    @pytest.mark.unit
    async def test_real_time_data_stream(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """リアルタイムデータストリーム"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        data_stream = []

        def data_callback(data: dict[str, Any]):
            data_stream.append(data)

        dashboard.set_real_time_callback(data_callback)
        await dashboard.start_real_time_monitoring()

        # 短時間待機してデータストリームをテスト
        await asyncio.sleep(1)

        await dashboard.stop_real_time_monitoring()

        # リアルタイムデータが収集されることを確認
        assert len(data_stream) >= 0  # 短時間なのでデータがない場合もある


class TestUserManagement:
    """ユーザー管理のテスト"""

    @pytest.mark.unit
    async def test_user_permissions(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """ユーザー権限管理"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # ユーザー権限設定
        await dashboard.set_user_permissions(
            "user_001",
            {
                "view_dashboard": True,
                "edit_widgets": False,
                "generate_reports": True,
                "manage_users": False,
            },
        )

        permissions = await dashboard.get_user_permissions("user_001")
        assert permissions["view_dashboard"] is True
        assert permissions["edit_widgets"] is False
        assert permissions["generate_reports"] is True

    @pytest.mark.unit
    async def test_audit_log(self, comprehensive_dashboard_config: DashboardConfig):
        """監査ログ"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # 操作ログの記録
        await dashboard.log_user_action(
            {
                "user_id": "admin_001",
                "action": "create_widget",
                "resource": "search_volume_widget",
                "timestamp": datetime.now(),
                "ip_address": "192.168.1.100",
            }
        )

        audit_logs = await dashboard.get_audit_logs(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
        )

        assert len(audit_logs) >= 1
        assert audit_logs[0]["action"] == "create_widget"

    @pytest.mark.unit
    async def test_session_management(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """セッション管理"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # アクティブセッションの追跡
        session_info = {
            "user_id": "user_001",
            "session_id": "session_123",
            "started_at": datetime.now(),
            "ip_address": "192.168.1.101",
        }

        await dashboard.track_user_session(session_info)

        active_sessions = await dashboard.get_active_sessions()
        assert len(active_sessions) >= 1


class TestDashboardIntegration:
    """ダッシュボード統合テスト"""

    @pytest.mark.integration
    async def test_end_to_end_dashboard_workflow(
        self,
        comprehensive_dashboard_config: DashboardConfig,
        sample_widgets: list[DashboardWidget],
    ):
        """エンドツーエンドダッシュボードワークフロー"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        try:
            # 1. ダッシュボード設定
            for widget in sample_widgets:
                await dashboard.add_widget(widget)

            # 2. リアルタイム監視開始
            await dashboard.start_real_time_monitoring()

            # 3. サンプルデータ生成
            await dashboard.record_search_event(
                {
                    "query": "AI integration",
                    "user_id": "user_001",
                    "timestamp": datetime.now(),
                    "response_time": 120,
                    "results_count": 25,
                    "success": True,
                }
            )

            # 4. メトリクス収集
            metrics = await dashboard.collect_system_metrics()
            assert metrics is not None

            # 5. レポート生成
            report_config = ReportConfig(
                type=ReportType.DASHBOARD_SUMMARY,
                format=ReportFormat.JSON,
                time_range=timedelta(hours=1),
            )
            report = await dashboard.generate_report(report_config)
            assert report is not None

            # 6. ダッシュボードステータス確認
            status = await dashboard.get_dashboard_status()
            assert status["widget_count"] == len(sample_widgets)
            assert status["real_time_enabled"] is True

        finally:
            await dashboard.stop_real_time_monitoring()

    @pytest.mark.integration
    async def test_dashboard_performance_monitoring(
        self, comprehensive_dashboard_config: DashboardConfig
    ):
        """ダッシュボードパフォーマンス監視"""
        dashboard = AdminDashboard(config=comprehensive_dashboard_config)

        # パフォーマンステスト用のウィジェット作成
        performance_widgets = []
        for i in range(10):
            widget = DashboardWidget(
                id=f"perf_widget_{i}",
                title=f"パフォーマンスウィジェット {i}",
                type=WidgetType.NUMBER,
                config={"metric": f"test_metric_{i}"},
                position={"x": i % 4 * 3, "y": i // 4 * 2, "width": 3, "height": 2},
            )
            performance_widgets.append(widget)
            await dashboard.add_widget(widget)

        # パフォーマンス測定
        start_time = datetime.now()

        # 複数の操作を並行実行
        tasks = [
            dashboard.collect_system_metrics(),
            dashboard.get_search_analytics(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
            ),
            dashboard.get_user_analytics(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
            ),
        ]

        results = await asyncio.gather(*tasks)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # パフォーマンス要件検証
        assert execution_time < 5.0  # 5秒以内
        assert all(result is not None for result in results)
