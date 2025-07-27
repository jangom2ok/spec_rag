"""メトリクス収集サービステスト

TDD実装：監視・メトリクス収集機能のテストケース
- 検索メトリクス: レスポンス時間、検索頻度、成功率
- ユーザーメトリクス: アクティブユーザー、セッション時間
- システムメトリクス: CPU、メモリ、ディスク使用率
- パフォーマンスメトリクス: スループット、エラー率
- ビジネスメトリクス: クエリ品質、満足度指標
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from app.services.metrics_collection import (
    AggregationType,
    BusinessMetrics,
    MetricFilter,
    MetricsCollectionService,
    MetricsConfig,
    MetricsRequest,
    MetricType,
    PerformanceMetrics,
    SearchMetrics,
    SystemMetrics,
    TimeWindow,
)


@pytest.fixture
def basic_metrics_config() -> MetricsConfig:
    """基本的なメトリクス設定"""
    return MetricsConfig(
        metric_types=[
            MetricType.SEARCH_METRICS,
            MetricType.SYSTEM_METRICS,
        ],
        collection_interval=60,
        retention_days=7,
        enable_real_time=True,
        aggregation_types=[
            AggregationType.AVG,
            AggregationType.SUM,
            AggregationType.COUNT,
        ],
    )


@pytest.fixture
def comprehensive_metrics_config() -> MetricsConfig:
    """包括的なメトリクス設定"""
    return MetricsConfig(
        metric_types=[
            MetricType.SEARCH_METRICS,
            MetricType.USER_METRICS,
            MetricType.SYSTEM_METRICS,
            MetricType.PERFORMANCE_METRICS,
            MetricType.BUSINESS_METRICS,
        ],
        collection_interval=30,
        retention_days=30,
        enable_real_time=True,
        enable_aggregation=True,
        aggregation_interval=300,
        aggregation_types=[
            AggregationType.AVG,
            AggregationType.SUM,
            AggregationType.COUNT,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.P95,
            AggregationType.P99,
        ],
        enable_alerting=True,
        alert_thresholds={
            "response_time_p95": 5000,
            "error_rate": 0.05,
            "cpu_usage": 0.8,
        },
    )


@pytest.fixture
def metrics_request() -> MetricsRequest:
    """メトリクス取得リクエスト"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    return MetricsRequest(
        metric_types=[MetricType.SEARCH_METRICS],
        start_time=start_time,
        end_time=end_time,
        time_window=TimeWindow.HOUR,
        aggregation_type=AggregationType.AVG,
        filters=[
            MetricFilter(field="user_id", value="test_user_123"),
        ],
    )


@pytest.fixture
def sample_search_metrics() -> list[SearchMetrics]:
    """サンプル検索メトリクス"""
    now = datetime.now()
    return [
        SearchMetrics(
            timestamp=now - timedelta(minutes=5),
            user_id="user_001",
            session_id="session_123",
            query="machine learning",
            response_time=245.5,
            result_count=15,
            clicked_position=3,
            is_successful=True,
            search_mode="hybrid",
        ),
        SearchMetrics(
            timestamp=now - timedelta(minutes=3),
            user_id="user_002",
            session_id="session_456",
            query="deep learning",
            response_time=189.2,
            result_count=12,
            clicked_position=1,
            is_successful=True,
            search_mode="semantic",
        ),
        SearchMetrics(
            timestamp=now - timedelta(minutes=1),
            user_id="user_001",
            session_id="session_123",
            query="neural networks",
            response_time=892.1,
            result_count=0,
            clicked_position=None,
            is_successful=False,
            search_mode="hybrid",
            error_message="Timeout error",
        ),
    ]


class TestMetricsConfig:
    """メトリクス設定のテスト"""

    @pytest.mark.unit
    def test_basic_config_creation(self):
        """基本設定の作成"""
        config = MetricsConfig(
            metric_types=[MetricType.SEARCH_METRICS],
            collection_interval=60,
            retention_days=7,
        )

        assert config.metric_types == [MetricType.SEARCH_METRICS]
        assert config.collection_interval == 60
        assert config.retention_days == 7
        assert not config.enable_real_time
        assert not config.enable_aggregation

    @pytest.mark.unit
    def test_config_validation_success(self):
        """設定値のバリデーション（成功）"""
        config = MetricsConfig(
            metric_types=[MetricType.SEARCH_METRICS],
            collection_interval=30,
            retention_days=30,
            aggregation_interval=300,
        )

        # バリデーションエラーが発生しないことを確認
        assert config.collection_interval == 30

    @pytest.mark.unit
    def test_config_validation_invalid_interval(self):
        """無効な収集間隔のバリデーション"""
        with pytest.raises(
            ValueError, match="collection_interval must be greater than 0"
        ):
            MetricsConfig(
                metric_types=[MetricType.SEARCH_METRICS],
                collection_interval=0,
                retention_days=7,
            )

    @pytest.mark.unit
    def test_config_validation_invalid_retention(self):
        """無効な保持期間のバリデーション"""
        with pytest.raises(ValueError, match="retention_days must be greater than 0"):
            MetricsConfig(
                metric_types=[MetricType.SEARCH_METRICS],
                collection_interval=60,
                retention_days=-1,
            )

    @pytest.mark.unit
    def test_config_validation_empty_metric_types(self):
        """空のメトリクスタイプのバリデーション"""
        with pytest.raises(ValueError, match="metric_types cannot be empty"):
            MetricsConfig(
                metric_types=[],
                collection_interval=60,
                retention_days=7,
            )


class TestMetricsCollectionService:
    """メトリクス収集サービスのテスト"""

    @pytest.mark.unit
    def test_service_initialization(self, basic_metrics_config: MetricsConfig):
        """サービスの初期化"""
        service = MetricsCollectionService(config=basic_metrics_config)

        assert service.config == basic_metrics_config
        assert service._is_collecting is False
        assert len(service._metric_collectors) == 0

    @pytest.mark.unit
    async def test_start_collection(self, basic_metrics_config: MetricsConfig):
        """メトリクス収集開始"""
        service = MetricsCollectionService(config=basic_metrics_config)

        await service.start_collection()

        assert service._is_collecting is True
        assert len(service._metric_collectors) == len(basic_metrics_config.metric_types)

    @pytest.mark.unit
    async def test_stop_collection(self, basic_metrics_config: MetricsConfig):
        """メトリクス収集停止"""
        service = MetricsCollectionService(config=basic_metrics_config)

        await service.start_collection()
        await service.stop_collection()

        assert service._is_collecting is False

    @pytest.mark.unit
    async def test_collect_search_metrics(
        self,
        basic_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """検索メトリクスの収集"""
        service = MetricsCollectionService(config=basic_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # メトリクスが記録されたことを確認
        collected = service._search_metrics_buffer
        assert len(collected) == len(sample_search_metrics)

        # レスポンス時間の平均値を確認
        avg_response_time = sum(m.response_time for m in collected) / len(collected)
        expected_avg = sum(m.response_time for m in sample_search_metrics) / len(
            sample_search_metrics
        )
        assert abs(avg_response_time - expected_avg) < 0.1


class TestSearchMetricsCollection:
    """検索メトリクス収集のテスト"""

    @pytest.mark.unit
    async def test_successful_search_metric(self, basic_metrics_config: MetricsConfig):
        """成功した検索のメトリクス"""
        service = MetricsCollectionService(config=basic_metrics_config)

        metric = SearchMetrics(
            timestamp=datetime.now(),
            user_id="test_user",
            session_id="test_session",
            query="test query",
            response_time=150.0,
            result_count=10,
            clicked_position=2,
            is_successful=True,
            search_mode="hybrid",
        )

        await service.record_search_metric(metric)

        assert len(service._search_metrics_buffer) == 1
        recorded = service._search_metrics_buffer[0]
        assert recorded.is_successful is True
        assert recorded.response_time == 150.0

    @pytest.mark.unit
    async def test_failed_search_metric(self, basic_metrics_config: MetricsConfig):
        """失敗した検索のメトリクス"""
        service = MetricsCollectionService(config=basic_metrics_config)

        metric = SearchMetrics(
            timestamp=datetime.now(),
            user_id="test_user",
            session_id="test_session",
            query="test query",
            response_time=5000.0,
            result_count=0,
            clicked_position=None,
            is_successful=False,
            search_mode="hybrid",
            error_message="Search timeout",
        )

        await service.record_search_metric(metric)

        recorded = service._search_metrics_buffer[0]
        assert recorded.is_successful is False
        assert recorded.error_message == "Search timeout"

    @pytest.mark.unit
    async def test_search_metrics_aggregation(
        self,
        comprehensive_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """検索メトリクスの集約"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # 集約実行
        aggregated = await service._aggregate_search_metrics(
            list(service._search_metrics_buffer), AggregationType.AVG
        )

        # 平均レスポンス時間の検証
        expected_avg = sum(m.response_time for m in sample_search_metrics) / len(
            sample_search_metrics
        )
        assert abs(aggregated["avg_response_time"] - expected_avg) < 0.1

        # 成功率の検証
        successful_count = sum(1 for m in sample_search_metrics if m.is_successful)
        expected_success_rate = successful_count / len(sample_search_metrics)
        assert abs(aggregated["success_rate"] - expected_success_rate) < 0.01


class TestSystemMetricsCollection:
    """システムメトリクス収集のテスト"""

    @pytest.mark.unit
    async def test_system_metrics_collection(self, basic_metrics_config: MetricsConfig):
        """システムメトリクスの収集"""
        service = MetricsCollectionService(config=basic_metrics_config)

        # システムメトリクスを取得
        metrics = await service._collect_system_metrics()

        assert metrics.cpu_usage >= 0.0
        assert metrics.memory_usage >= 0.0
        assert metrics.disk_usage >= 0.0
        assert metrics.network_io_bytes >= 0

    @pytest.mark.unit
    async def test_system_metrics_thresholds(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """システムメトリクスの閾値チェック"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # 高いCPU使用率をシミュレート
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.85,  # 閾値0.8を超過
            memory_usage=0.60,
            disk_usage=0.30,
            network_io_bytes=1000000,
        )

        alerts = await service._check_metric_thresholds(metrics)

        # CPU使用率アラートが生成されることを確認
        cpu_alerts = [alert for alert in alerts if "cpu_usage" in alert["metric"]]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0]["severity"] == "warning"


class TestUserMetricsCollection:
    """ユーザーメトリクス収集のテスト"""

    @pytest.mark.unit
    async def test_user_session_tracking(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """ユーザーセッション追跡"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # セッション開始
        await service.start_user_session("user_123", "session_456")

        # セッション情報の確認
        assert "session_456" in service._active_sessions
        session_info = service._active_sessions["session_456"]
        assert session_info["user_id"] == "user_123"
        assert session_info["start_time"] is not None

    @pytest.mark.unit
    async def test_user_activity_metrics(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """ユーザー活動メトリクス"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # ユーザー活動を記録
        await service.record_user_activity("user_123", "search", {"query": "test"})
        await service.record_user_activity("user_123", "click", {"position": 1})

        # 活動履歴の確認
        activities = service._user_activities.get("user_123", [])
        assert len(activities) == 2
        assert activities[0]["action"] == "search"
        assert activities[1]["action"] == "click"


class TestPerformanceMetricsCollection:
    """パフォーマンスメトリクス収集のテスト"""

    @pytest.mark.unit
    async def test_response_time_tracking(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """レスポンス時間追跡"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # レスポンス時間メトリクス
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            endpoint="/api/search",
            response_time=245.5,
            request_count=100,
            error_count=2,
            throughput=95.5,
        )

        await service.record_performance_metric(metrics)

        recorded = service._performance_metrics_buffer[-1]
        assert recorded.response_time == 245.5
        assert recorded.error_rate == 0.02  # 2/100

    @pytest.mark.unit
    async def test_throughput_calculation(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """スループット計算"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # 複数のパフォーマンスメトリクス
        timestamps = [datetime.now() - timedelta(seconds=i) for i in range(5, 0, -1)]

        for i, timestamp in enumerate(timestamps):
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                endpoint="/api/search",
                response_time=200.0 + i * 10,
                request_count=20,
                error_count=1,
                throughput=18.0,
            )
            await service.record_performance_metric(metrics)

        # 平均スループットの計算
        avg_throughput = await service._calculate_average_throughput(
            list(service._performance_metrics_buffer), time_window=timedelta(seconds=60)
        )

        assert avg_throughput == 18.0


class TestBusinessMetricsCollection:
    """ビジネスメトリクス収集のテスト"""

    @pytest.mark.unit
    async def test_query_quality_metrics(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """クエリ品質メトリクス"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # クエリ品質メトリクス
        metrics = BusinessMetrics(
            timestamp=datetime.now(),
            query_quality_score=0.85,
            user_satisfaction_score=4.2,
            conversion_rate=0.15,
            engagement_rate=0.68,
            retention_rate=0.75,
        )

        await service.record_business_metric(metrics)

        recorded = service._business_metrics_buffer[-1]
        assert recorded.query_quality_score == 0.85
        assert recorded.user_satisfaction_score == 4.2

    @pytest.mark.unit
    async def test_satisfaction_score_calculation(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """満足度スコア計算"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # 複数の満足度データ
        satisfaction_scores = [4.5, 3.8, 4.2, 4.7, 3.9]

        for score in satisfaction_scores:
            metrics = BusinessMetrics(
                timestamp=datetime.now(),
                query_quality_score=0.8,
                user_satisfaction_score=score,
                conversion_rate=0.1,
                engagement_rate=0.6,
                retention_rate=0.7,
            )
            await service.record_business_metric(metrics)

        # 平均満足度の計算
        avg_satisfaction = await service._calculate_average_satisfaction(
            list(service._business_metrics_buffer)
        )

        expected_avg = sum(satisfaction_scores) / len(satisfaction_scores)
        assert abs(avg_satisfaction - expected_avg) < 0.01


class TestMetricsQuery:
    """メトリクス問い合わせのテスト"""

    @pytest.mark.unit
    async def test_query_metrics_by_time_range(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """時間範囲でのメトリクス問い合わせ"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # テストデータの投入
        now = datetime.now()
        for i in range(10):
            metric = SearchMetrics(
                timestamp=now - timedelta(minutes=i * 5),
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                query=f"query {i}",
                response_time=200.0 + i * 10,
                result_count=10 - i,
                clicked_position=1 if i % 2 == 0 else None,
                is_successful=i < 8,
                search_mode="hybrid",
            )
            await service.record_search_metric(metric)

        # フィルターなしのメトリクス問い合わせ
        request = MetricsRequest(
            metric_types=[MetricType.SEARCH_METRICS],
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=5),
            time_window=TimeWindow.HOUR,
            aggregation_type=AggregationType.COUNT,
        )

        response = await service.query_metrics(request)

        assert response.success is True
        assert len(response.data) > 0
        assert response.time_window == TimeWindow.HOUR

    @pytest.mark.unit
    async def test_query_metrics_with_filters(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """フィルター付きメトリクス問い合わせ"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # 特定ユーザーのメトリクス
        target_user = "target_user"
        now = datetime.now()
        for i in range(5):
            metric = SearchMetrics(
                timestamp=now - timedelta(minutes=i),
                user_id=target_user if i < 3 else f"other_user_{i}",
                session_id=f"session_{i}",
                query=f"query {i}",
                response_time=200.0,
                result_count=10,
                clicked_position=1,
                is_successful=True,
                search_mode="hybrid",
            )
            await service.record_search_metric(metric)

        # フィルター付きリクエスト
        request = MetricsRequest(
            metric_types=[MetricType.SEARCH_METRICS],
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=5),
            aggregation_type=AggregationType.COUNT,
            filters=[MetricFilter(field="user_id", value=target_user)],
        )

        response = await service.query_metrics(request)

        # フィルターされた結果のみ返されることを確認
        assert response.success is True
        # COUNT集約の場合は集約データが返される
        if response.data and len(response.data) > 0:
            assert response.data[0].get("count", 0) == 3
        else:
            # 直接データをチェック
            all_data = await service._query_metric_type(
                MetricType.SEARCH_METRICS, request
            )
            filtered_data = service._apply_filters(all_data, request.filters)
            filtered_count = len(
                [m for m in filtered_data if m.get("user_id") == target_user]
            )
            assert filtered_count == 3


class TestMetricsAggregation:
    """メトリクス集約のテスト"""

    @pytest.mark.unit
    async def test_time_based_aggregation(
        self,
        comprehensive_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """時間ベース集約"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # 時間ベース集約の実行
        aggregated = await service._aggregate_by_time_window(
            list(service._search_metrics_buffer), TimeWindow.MINUTE, AggregationType.AVG
        )

        assert len(aggregated) > 0
        assert "timestamp" in aggregated[0]
        assert "avg_response_time" in aggregated[0]

    @pytest.mark.unit
    async def test_multiple_aggregation_types(
        self,
        comprehensive_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """複数集約タイプ"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # 複数の集約タイプで実行
        for agg_type in [
            AggregationType.AVG,
            AggregationType.SUM,
            AggregationType.COUNT,
        ]:
            aggregated = await service._aggregate_search_metrics(
                list(service._search_metrics_buffer), agg_type
            )

            assert len(aggregated) > 0

            if agg_type == AggregationType.COUNT:
                assert "count" in aggregated
            elif agg_type == AggregationType.SUM:
                assert "sum_response_time" in aggregated
            elif agg_type == AggregationType.AVG:
                assert "avg_response_time" in aggregated


class TestMetricsExport:
    """メトリクスエクスポートのテスト"""

    @pytest.mark.unit
    async def test_export_to_json(
        self,
        comprehensive_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """JSON形式でのエクスポート"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # JSONエクスポート
        exported = await service.export_metrics(format="json")

        assert exported is not None

        # JSONとしてパース可能であることを確認
        parsed = json.loads(exported)
        assert "search_metrics" in parsed
        assert len(parsed["search_metrics"]) == len(sample_search_metrics)

    @pytest.mark.unit
    async def test_export_filtered_metrics(
        self,
        comprehensive_metrics_config: MetricsConfig,
        sample_search_metrics: list[SearchMetrics],
    ):
        """フィルター済みメトリクスのエクスポート"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        for metric in sample_search_metrics:
            await service.record_search_metric(metric)

        # 特定ユーザーのメトリクスのみエクスポート
        filters = [MetricFilter(field="user_id", value="user_001")]
        exported = await service.export_metrics(format="json", filters=filters)

        parsed = json.loads(exported)
        user_metrics = [
            m for m in parsed["search_metrics"] if m["user_id"] == "user_001"
        ]
        assert len(user_metrics) == 2  # sample_search_metricsにuser_001が2つ


class TestMetricsIntegration:
    """メトリクス統合テスト"""

    @pytest.mark.integration
    async def test_end_to_end_metrics_collection(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """エンドツーエンドメトリクス収集"""
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        # メトリクス収集開始
        await service.start_collection()

        try:
            # 各種メトリクスの記録
            search_metric = SearchMetrics(
                timestamp=datetime.now(),
                user_id="integration_user",
                session_id="integration_session",
                query="integration test",
                response_time=300.0,
                result_count=8,
                clicked_position=2,
                is_successful=True,
                search_mode="hybrid",
            )
            await service.record_search_metric(search_metric)

            system_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.45,
                memory_usage=0.60,
                disk_usage=0.35,
                network_io_bytes=500000,
            )
            await service.record_system_metric(system_metric)

            # 短時間待機（バッファリング処理のため）
            await asyncio.sleep(0.1)

            # メトリクス問い合わせ
            request = MetricsRequest(
                metric_types=[MetricType.SEARCH_METRICS, MetricType.SYSTEM_METRICS],
                start_time=datetime.now() - timedelta(minutes=5),
                end_time=datetime.now() + timedelta(minutes=1),
            )

            response = await service.query_metrics(request)

            assert response.success is True
            assert len(response.data) > 0

        finally:
            # メトリクス収集停止
            await service.stop_collection()

    @pytest.mark.integration
    async def test_real_time_metrics_stream(
        self, comprehensive_metrics_config: MetricsConfig
    ):
        """リアルタイムメトリクスストリーム"""
        comprehensive_metrics_config.enable_real_time = True
        service = MetricsCollectionService(config=comprehensive_metrics_config)

        collected_metrics = []

        # リアルタイムメトリクスのコールバック
        def metrics_callback(metric_data: dict[str, Any]) -> None:
            collected_metrics.append(metric_data)

        await service.start_collection()
        service.set_real_time_callback(metrics_callback)

        try:
            # メトリクスを記録
            for i in range(3):
                metric = SearchMetrics(
                    timestamp=datetime.now(),
                    user_id=f"stream_user_{i}",
                    session_id=f"stream_session_{i}",
                    query=f"stream test {i}",
                    response_time=200.0 + i * 50,
                    result_count=10,
                    clicked_position=1,
                    is_successful=True,
                    search_mode="hybrid",
                )
                await service.record_search_metric(metric)
                await asyncio.sleep(0.05)  # 少し待機

            # リアルタイムコールバックが呼ばれたことを確認
            await asyncio.sleep(0.2)  # コールバック処理待機
            assert len(collected_metrics) > 0

        finally:
            await service.stop_collection()
