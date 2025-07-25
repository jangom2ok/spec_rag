"""
Comprehensive test coverage for service files with missing coverage.
This file covers:
- app/services/document_collector.py
- app/services/embedding_tasks.py
- app/services/reranker.py
- app/services/query_expansion.py
- app/services/document_chunker.py
- app/services/admin_dashboard.py
- app/services/alerting_service.py
- app/services/search_suggestions.py
- app/services/search_diversity.py
- app/services/hybrid_search_engine.py
- app/services/logging_analysis.py
- app/services/metrics_collection.py
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest


class TestDocumentCollectorCoverage:
    """Test missing coverage in document_collector.py."""

    @pytest.mark.asyncio
    async def test_collect_from_github(self):
        """Test collecting documents from GitHub."""
        from app.services.document_collector import DocumentCollector

        collector = DocumentCollector()

        # Mock GitHub API response
        mock_response = Mock()
        mock_response.json = AsyncMock(
            return_value={
                "items": [
                    {
                        "name": "README.md",
                        "path": "README.md",
                        "download_url": "https://raw.github.com/test/README.md",
                        "type": "file",
                    }
                ]
            }
        )
        mock_response.status = 200

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            documents = await collector.collect_from_github(
                "https://github.com/test/repo"
            )

            assert len(documents) > 0

    @pytest.mark.asyncio
    async def test_collect_from_confluence(self):
        """Test collecting documents from Confluence."""
        from app.services.document_collector import DocumentCollector

        collector = DocumentCollector()

        # Mock Confluence API
        with patch("app.services.document_collector.Confluence") as mock_confluence:
            mock_instance = Mock()
            mock_confluence.return_value = mock_instance

            mock_instance.get_all_spaces.return_value = [
                {"key": "TEST", "name": "Test Space"}
            ]
            mock_instance.get_all_pages_from_space.return_value = [
                {
                    "id": "123",
                    "title": "Test Page",
                    "body": {"storage": {"value": "<p>Test content</p>"}},
                }
            ]

            documents = await collector.collect_from_confluence(
                "https://test.atlassian.net", "user@test.com", "api-token"
            )

            assert len(documents) > 0

    @pytest.mark.asyncio
    async def test_collect_from_local_files(self):
        """Test collecting documents from local files."""
        from app.services.document_collector import DocumentCollector

        collector = DocumentCollector()

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("pathlib.Path.is_dir") as mock_is_dir:
                mock_is_dir.return_value = True

                with patch("pathlib.Path.rglob") as mock_rglob:
                    mock_file = Mock()
                    mock_file.is_file.return_value = True
                    mock_file.suffix = ".txt"
                    mock_file.name = "test.txt"
                    mock_file.read_text.return_value = "Test content"

                    mock_rglob.return_value = [mock_file]

                    documents = await collector.collect_from_local("/test/path")

                    assert len(documents) > 0

    @pytest.mark.asyncio
    async def test_process_markdown_with_metadata(self):
        """Test processing markdown with metadata."""
        from app.services.document_collector import DocumentCollector

        collector = DocumentCollector()

        markdown_content = """---
title: Test Document
author: Test Author
tags: [test, example]
---

# Test Content

This is a test document.
"""

        document = collector._process_markdown(markdown_content, "test.md")

        assert document["title"] == "Test Document"
        assert document["metadata"]["author"] == "Test Author"
        assert "test" in document["metadata"]["tags"]


class TestEmbeddingTasksCoverage:
    """Test missing coverage in embedding_tasks.py."""

    @pytest.mark.asyncio
    async def test_process_document_batch_with_errors(self):
        """Test processing document batch with errors."""
        from app.services.embedding_tasks import process_document_batch

        documents = [
            {"id": "1", "content": "Test content 1"},
            {"id": "2", "content": "Test content 2"},
        ]

        with patch("app.services.embedding_tasks.EmbeddingService") as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance

            # First call succeeds, second fails
            mock_instance.generate_embeddings = AsyncMock(
                side_effect=[
                    {"dense": [[0.1, 0.2]], "sparse": [[0.3, 0.4]]},
                    Exception("Embedding error"),
                ]
            )

            result = await process_document_batch(documents)

            assert result["success"] is False
            assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_update_embeddings_task(self):
        """Test update embeddings task."""
        from app.services.embedding_tasks import update_embeddings_for_documents

        document_ids = ["doc1", "doc2"]

        with patch("app.services.embedding_tasks.get_documents_by_ids") as mock_get:
            mock_get.return_value = [
                {"id": "doc1", "content": "Content 1"},
                {"id": "doc2", "content": "Content 2"},
            ]

            with patch(
                "app.services.embedding_tasks.process_document_batch"
            ) as mock_process:
                mock_process.return_value = {"success": True, "processed": 2}

                result = await update_embeddings_for_documents(document_ids)

                assert result["processed"] == 2

    def test_celery_task_wrapper(self):
        """Test Celery task wrapper functionality."""
        from app.services.embedding_tasks import generate_embeddings_task

        with patch("app.services.embedding_tasks.celery_app"):
            # Test task registration
            assert generate_embeddings_task is not None


class TestRerankerCoverage:
    """Test missing coverage in reranker.py."""

    @pytest.mark.asyncio
    async def test_rerank_with_cross_encoder(self):
        """Test reranking with cross-encoder model."""
        from app.services.reranker import Reranker

        reranker = Reranker(model_name="ms-marco-MiniLM-L-6-v2")

        query = "test query"
        documents = [
            {"content": "Relevant document", "search_score": 0.8},
            {"content": "Less relevant", "search_score": 0.6},
        ]

        with patch("app.services.reranker.CrossEncoder") as mock_encoder:
            mock_model = Mock()
            mock_encoder.return_value = mock_model
            mock_model.predict.return_value = [0.9, 0.4]

            reranker.model = mock_model
            reranker.model_type = "cross-encoder"

            result = await reranker.rerank(query, documents)

            assert result[0]["rerank_score"] == 0.9
            assert result[0]["content"] == "Relevant document"

    @pytest.mark.asyncio
    async def test_rerank_with_insufficient_documents(self):
        """Test reranking with few documents."""
        from app.services.reranker import Reranker

        reranker = Reranker()

        # Test with empty documents
        result = await reranker.rerank("query", [])
        assert result == []

        # Test with single document
        single_doc = [{"content": "Only doc", "search_score": 0.5}]
        result = await reranker.rerank("query", single_doc)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_batch_rerank(self):
        """Test batch reranking functionality."""
        from app.services.reranker import Reranker

        reranker = Reranker()

        queries = ["query1", "query2"]
        document_batches = [
            [{"content": "Doc1", "search_score": 0.7}],
            [{"content": "Doc2", "search_score": 0.8}],
        ]

        with patch.object(reranker, "rerank") as mock_rerank:
            mock_rerank.side_effect = [
                [{"content": "Doc1", "rerank_score": 0.9}],
                [{"content": "Doc2", "rerank_score": 0.85}],
            ]

            results = await reranker.batch_rerank(queries, document_batches)

            assert len(results) == 2

    def test_reranker_initialization_errors(self):
        """Test reranker initialization with errors."""
        from app.services.reranker import Reranker

        with patch("app.services.reranker.CrossEncoder") as mock_encoder:
            mock_encoder.side_effect = Exception("Model loading failed")

            reranker = Reranker(model_name="invalid-model")
            assert reranker.model is None


class TestQueryExpansionCoverage:
    """Test missing coverage in query_expansion.py."""

    @pytest.mark.asyncio
    async def test_expand_query_with_synonyms(self):
        """Test query expansion with synonyms."""
        from app.services.query_expansion import QueryExpander

        expander = QueryExpander()

        # Mock WordNet
        with patch("app.services.query_expansion.wordnet") as mock_wn:
            mock_synset = Mock()
            mock_synset.lemmas.return_value = [
                Mock(name=lambda: "test"),
                Mock(name=lambda: "exam"),
                Mock(name=lambda: "trial"),
            ]
            mock_wn.synsets.return_value = [mock_synset]

            expanded = await expander.expand_with_synonyms("test")

            assert "exam" in expanded
            assert "trial" in expanded

    @pytest.mark.asyncio
    async def test_expand_query_with_embeddings(self):
        """Test query expansion with embeddings."""
        from app.services.query_expansion import QueryExpander

        expander = QueryExpander()

        with patch.object(expander, "embedding_service") as mock_service:
            mock_service.generate_embeddings.return_value = {"dense": [[0.1, 0.2, 0.3]]}

            with patch.object(expander, "_find_similar_terms") as mock_similar:
                mock_similar.return_value = ["related1", "related2"]

                expanded = await expander.expand_with_embeddings("test query")

                assert len(expanded) > 0

    @pytest.mark.asyncio
    async def test_contextual_expansion(self):
        """Test contextual query expansion."""
        from app.services.query_expansion import QueryExpander

        expander = QueryExpander()

        context = {
            "previous_queries": ["python programming", "data science"],
            "domain": "technology",
        }

        expanded = await expander.expand_with_context("machine learning", context)

        assert isinstance(expanded, dict)
        assert "expanded_terms" in expanded

    def test_query_expansion_caching(self):
        """Test query expansion caching mechanism."""
        from app.services.query_expansion import QueryExpander

        expander = QueryExpander()

        # First call
        with patch.object(expander, "_expand_internal") as mock_expand:
            mock_expand.return_value = ["term1", "term2"]

            expander.expand_with_cache("test")
            expander.expand_with_cache("test")

            # Should only call once due to caching
            mock_expand.assert_called_once()


class TestDocumentChunkerCoverage:
    """Test missing coverage in document_chunker.py."""

    @pytest.mark.asyncio
    async def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        from app.services.document_chunker import DocumentChunker

        chunker = DocumentChunker(strategy="semantic")

        document = {
            "content": "First paragraph about topic A. " * 50
            + "\n\n"
            + "Second paragraph about topic B. " * 50,
            "id": "doc1",
        }

        with patch.object(chunker, "_calculate_semantic_similarity") as mock_sim:
            mock_sim.return_value = 0.3  # Low similarity indicates boundary

            chunks = await chunker.chunk_document(document)

            assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_sliding_window_chunking(self):
        """Test sliding window chunking."""
        from app.services.document_chunker import DocumentChunker

        chunker = DocumentChunker(
            strategy="sliding_window", chunk_size=100, overlap_size=20
        )

        document = {"content": "Test content. " * 100, "id": "doc1"}

        chunks = await chunker.chunk_document(document)

        # Check overlap
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i]["content"][-20:]
            chunks[i + 1]["content"][:20]
            assert chunk1_end in chunks[i + 1]["content"]

    @pytest.mark.asyncio
    async def test_chunk_with_metadata_preservation(self):
        """Test chunking with metadata preservation."""
        from app.services.document_chunker import DocumentChunker

        chunker = DocumentChunker()

        document = {
            "content": "Test content " * 200,
            "id": "doc1",
            "metadata": {
                "author": "Test Author",
                "date": "2024-01-01",
                "tags": ["test", "example"],
            },
        }

        chunks = await chunker.chunk_document(document)

        for chunk in chunks:
            assert chunk["metadata"]["author"] == "Test Author"
            assert "test" in chunk["metadata"]["tags"]

    @pytest.mark.asyncio
    async def test_hierarchical_chunking(self):
        """Test hierarchical chunking with structure detection."""
        from app.services.document_chunker import DocumentChunker

        chunker = DocumentChunker(strategy="hierarchical")

        document = {
            "content": """
# Main Title

## Section 1
Content for section 1.

### Subsection 1.1
Details about subsection.

## Section 2
Content for section 2.
""",
            "id": "doc1",
        }

        chunks = await chunker.chunk_document(document)

        # Should detect hierarchical structure
        assert any("hierarchy_level" in chunk.get("metadata", {}) for chunk in chunks)


class TestAdminDashboardCoverage:
    """Test missing coverage in admin_dashboard.py."""

    @pytest.mark.asyncio
    async def test_collect_system_metrics_error_handling(self):
        """Test system metrics collection with errors."""
        from app.services.admin_dashboard import AdminDashboard

        dashboard = AdminDashboard()

        with patch("psutil.cpu_percent") as mock_cpu:
            mock_cpu.side_effect = Exception("CPU error")

            metrics = await dashboard.collect_system_metrics()

            # Should handle error gracefully
            assert "error" in metrics or metrics["cpu_usage"] == 0

    @pytest.mark.asyncio
    async def test_generate_usage_report(self):
        """Test usage report generation."""
        from app.services.admin_dashboard import AdminDashboard

        dashboard = AdminDashboard()

        with patch.object(dashboard, "_get_usage_data") as mock_usage:
            mock_usage.return_value = {
                "total_queries": 1000,
                "unique_users": 50,
                "average_response_time": 0.5,
            }

            report = await dashboard.generate_usage_report(
                start_date=datetime.now() - timedelta(days=7), end_date=datetime.now()
            )

            assert report["total_queries"] == 1000

    @pytest.mark.asyncio
    async def test_export_dashboard_data(self):
        """Test dashboard data export."""
        from app.services.admin_dashboard import AdminDashboard

        dashboard = AdminDashboard()

        with patch("builtins.open", mock_open()) as mock_file:
            await dashboard.export_dashboard_data(
                format="csv", output_path="/tmp/test_dashboard.csv"  # noqa: S108
            )

            mock_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_time_monitoring(self):
        """Test real-time monitoring functionality."""
        from app.services.admin_dashboard import AdminDashboard

        dashboard = AdminDashboard()

        # Start monitoring
        monitoring_task = asyncio.create_task(
            dashboard.start_real_time_monitoring(interval=0.1)
        )

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        dashboard.stop_monitoring = True
        await monitoring_task


class TestAlertingServiceCoverage:
    """Test missing coverage in alerting_service.py."""

    @pytest.mark.asyncio
    async def test_send_email_alert(self):
        """Test sending email alerts."""
        from app.services.alerting_service import AlertingService

        service = AlertingService()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            await service.send_email_alert(
                subject="Test Alert",
                body="Alert message",
                recipients=["admin@example.com"],
            )

            mock_server.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_slack_alert(self):
        """Test sending Slack alerts."""
        from app.services.alerting_service import AlertingService

        service = AlertingService()

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            await service.send_slack_alert(
                webhook_url="https://hooks.slack.com/test", message="Test alert"
            )

    @pytest.mark.asyncio
    async def test_check_alert_conditions(self):
        """Test alert condition checking."""
        from app.services.alerting_service import AlertingService

        service = AlertingService()

        # Define alert rules
        rules = [
            {
                "name": "High CPU",
                "condition": lambda metrics: metrics["cpu_usage"] > 80,
                "action": "email",
            },
            {
                "name": "Low Memory",
                "condition": lambda metrics: metrics["memory_available"] < 1000,
                "action": "slack",
            },
        ]

        metrics = {"cpu_usage": 85, "memory_available": 500}

        triggered = await service.check_conditions(rules, metrics)

        assert len(triggered) == 2

    @pytest.mark.asyncio
    async def test_alert_aggregation(self):
        """Test alert aggregation to prevent spam."""
        from app.services.alerting_service import AlertingService

        service = AlertingService()

        # Send multiple similar alerts
        for _i in range(10):
            await service.queue_alert(
                {
                    "type": "high_cpu",
                    "message": "CPU usage is high",
                    "severity": "warning",
                }
            )

        # Should aggregate into single alert
        aggregated = await service.get_aggregated_alerts()

        assert len(aggregated) <= 5  # Reasonable aggregation


class TestSearchSuggestionsCoverage:
    """Test missing coverage in search_suggestions.py."""

    @pytest.mark.asyncio
    async def test_generate_suggestions_from_history(self):
        """Test generating suggestions from search history."""
        from app.services.search_suggestions import SearchSuggestionsService

        service = SearchSuggestionsService()

        with patch.object(service, "_get_search_history") as mock_history:
            mock_history.return_value = [
                {"query": "python tutorial", "frequency": 10},
                {"query": "python pandas", "frequency": 8},
                {"query": "java basics", "frequency": 5},
            ]

            suggestions = await service.get_suggestions("pyth")

            assert "python tutorial" in suggestions
            assert "python pandas" in suggestions
            assert "java basics" not in suggestions

    @pytest.mark.asyncio
    async def test_personalized_suggestions(self):
        """Test personalized suggestions based on user profile."""
        from app.services.search_suggestions import SearchSuggestionsService

        service = SearchSuggestionsService()

        user_profile = {
            "interests": ["machine learning", "data science"],
            "recent_searches": ["neural networks", "deep learning"],
        }

        suggestions = await service.get_personalized_suggestions("learn", user_profile)

        # Should prioritize ML-related suggestions
        assert any("learning" in s.lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_typo_correction_suggestions(self):
        """Test suggestions with typo correction."""
        from app.services.search_suggestions import SearchSuggestionsService

        service = SearchSuggestionsService()

        with patch.object(service, "_correct_typo") as mock_correct:
            mock_correct.return_value = "python"

            await service.get_suggestions("pythn")

            mock_correct.assert_called_with("pythn")

    @pytest.mark.asyncio
    async def test_trending_suggestions(self):
        """Test trending topic suggestions."""
        from app.services.search_suggestions import SearchSuggestionsService

        service = SearchSuggestionsService()

        with patch.object(service, "_get_trending_topics") as mock_trending:
            mock_trending.return_value = [
                "ChatGPT integration",
                "Vector databases",
                "RAG systems",
            ]

            suggestions = await service.get_trending_suggestions()

            assert "ChatGPT integration" in suggestions


class TestSearchDiversityCoverage:
    """Test missing coverage in search_diversity.py."""

    @pytest.mark.asyncio
    async def test_mmr_diversification(self):
        """Test Maximal Marginal Relevance diversification."""
        from app.services.search_diversity import DiversityOptimizer

        optimizer = DiversityOptimizer(method="mmr")

        documents = [
            {"id": "1", "content": "Python programming basics", "score": 0.9},
            {"id": "2", "content": "Python programming advanced", "score": 0.85},
            {"id": "3", "content": "Java programming basics", "score": 0.8},
            {"id": "4", "content": "Machine learning with Python", "score": 0.75},
        ]

        with patch.object(optimizer, "_calculate_similarity") as mock_sim:
            # High similarity between Python docs
            mock_sim.side_effect = lambda d1, d2: (
                0.9 if "Python" in d1["content"] and "Python" in d2["content"] else 0.2
            )

            diverse_docs = await optimizer.diversify(documents, lambda_param=0.5)

            # Should include diverse topics
            contents = [d["content"] for d in diverse_docs]
            assert any("Java" in c for c in contents)

    @pytest.mark.asyncio
    async def test_temporal_diversification(self):
        """Test temporal diversification."""
        from app.services.search_diversity import DiversityOptimizer

        optimizer = DiversityOptimizer(method="temporal")

        documents = [
            {"id": "1", "date": "2024-01-01", "score": 0.9},
            {"id": "2", "date": "2024-01-02", "score": 0.85},
            {"id": "3", "date": "2023-06-01", "score": 0.8},
            {"id": "4", "date": "2022-01-01", "score": 0.75},
        ]

        diverse_docs = await optimizer.diversify_temporal(documents)

        # Should include documents from different time periods
        years = [d["date"][:4] for d in diverse_docs]
        assert len(set(years)) > 1

    @pytest.mark.asyncio
    async def test_source_diversification(self):
        """Test source-based diversification."""
        from app.services.search_diversity import DiversityOptimizer

        optimizer = DiversityOptimizer(method="source")

        documents = [
            {"id": "1", "source": "confluence", "score": 0.9},
            {"id": "2", "source": "confluence", "score": 0.85},
            {"id": "3", "source": "github", "score": 0.8},
            {"id": "4", "source": "jira", "score": 0.75},
        ]

        diverse_docs = await optimizer.diversify_by_source(documents, max_per_source=1)

        # Should limit documents per source
        source_counts = {}
        for doc in diverse_docs:
            source = doc["source"]
            source_counts[source] = source_counts.get(source, 0) + 1

        assert all(count <= 1 for count in source_counts.values())


class TestMetricsCollectionCoverage:
    """Test missing coverage in metrics_collection.py."""

    @pytest.mark.asyncio
    async def test_collect_search_metrics(self):
        """Test search metrics collection."""
        from app.services.metrics_collection import MetricsCollector

        collector = MetricsCollector()

        search_event = {
            "query": "test query",
            "results_count": 10,
            "response_time": 0.5,
            "user_id": "user123",
            "timestamp": datetime.now(),
        }

        await collector.record_search_event(search_event)

        metrics = await collector.get_search_metrics(
            start_date=datetime.now() - timedelta(hours=1), end_date=datetime.now()
        )

        assert metrics["total_searches"] >= 1

    @pytest.mark.asyncio
    async def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        from app.services.metrics_collection import MetricsCollector

        collector = MetricsCollector()

        # Record multiple events
        for i in range(100):
            await collector.record_event(
                {
                    "type": "search",
                    "response_time": 0.1 + (i % 10) * 0.1,
                    "timestamp": datetime.now(),
                }
            )

        aggregated = await collector.aggregate_metrics(
            metric_type="search", aggregation="percentile", percentiles=[50, 95, 99]
        )

        assert "p50" in aggregated
        assert "p95" in aggregated
        assert "p99" in aggregated

    @pytest.mark.asyncio
    async def test_export_metrics_to_prometheus(self):
        """Test exporting metrics to Prometheus format."""
        from app.services.metrics_collection import MetricsCollector

        collector = MetricsCollector()

        prometheus_metrics = await collector.export_prometheus_format()

        assert "# HELP" in prometheus_metrics
        assert "# TYPE" in prometheus_metrics

    @pytest.mark.asyncio
    async def test_metrics_alerting_integration(self):
        """Test metrics-based alerting."""
        from app.services.metrics_collection import MetricsCollector

        collector = MetricsCollector()

        # Define alert thresholds
        thresholds = {
            "high_response_time": {
                "metric": "response_time_p95",
                "operator": ">",
                "value": 1.0,
            },
            "low_success_rate": {
                "metric": "success_rate",
                "operator": "<",
                "value": 0.95,
            },
        }

        current_metrics = {"response_time_p95": 1.5, "success_rate": 0.9}

        alerts = await collector.check_thresholds(thresholds, current_metrics)

        assert len(alerts) == 2


class TestHybridSearchEngineCoverage:
    """Test missing coverage in hybrid_search_engine.py."""

    @pytest.mark.asyncio
    async def test_search_with_filters_and_facets(self):
        """Test search with complex filters and faceting."""
        from app.services.hybrid_search_engine import HybridSearchEngine

        engine = HybridSearchEngine()

        query = {
            "text": "machine learning",
            "filters": [
                {"field": "source_type", "value": "confluence", "operator": "eq"},
                {"field": "date", "value": "2024-01-01", "operator": "gte"},
            ],
            "facets": ["source_type", "language", "author"],
        }

        with patch.object(engine, "_execute_dense_search") as mock_dense:
            mock_dense.return_value = [
                {"id": "1", "score": 0.9, "source_type": "confluence"}
            ]

            with patch.object(engine, "_execute_sparse_search") as mock_sparse:
                mock_sparse.return_value = [
                    {"id": "2", "score": 0.8, "source_type": "github"}
                ]

                result = await engine.search(query)

                assert "facets" in result
                assert "source_type" in result["facets"]

    @pytest.mark.asyncio
    async def test_search_result_caching(self):
        """Test search result caching mechanism."""
        from app.services.hybrid_search_engine import HybridSearchEngine

        engine = HybridSearchEngine(enable_cache=True)

        query = {"text": "test query", "max_results": 10}

        with patch.object(engine, "_execute_search") as mock_search:
            mock_search.return_value = {
                "documents": [{"id": "1", "content": "Test"}],
                "total_hits": 1,
            }

            # First search
            result1 = await engine.search(query)

            # Second search (should use cache)
            result2 = await engine.search(query)

            # Should only call once
            mock_search.assert_called_once()
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_adaptive_weight_adjustment(self):
        """Test adaptive weight adjustment based on query type."""
        from app.services.hybrid_search_engine import HybridSearchEngine

        engine = HybridSearchEngine(adaptive_weights=True)

        # Technical query (should favor sparse)
        technical_query = {"text": "SQLException java.lang.NullPointerException"}
        weights = await engine._determine_weights(technical_query)
        assert weights["sparse"] > weights["dense"]

        # Natural language query (should favor dense)
        natural_query = {"text": "how to implement authentication in web app"}
        weights = await engine._determine_weights(natural_query)
        assert weights["dense"] > weights["sparse"]


class TestLoggingAnalysisCoverage:
    """Test missing coverage in logging_analysis.py."""

    @pytest.mark.asyncio
    async def test_parse_log_patterns(self):
        """Test log pattern parsing."""
        from app.services.logging_analysis import LogAnalyzer

        analyzer = LogAnalyzer()

        log_lines = [
            "2024-01-01 10:00:00 ERROR Failed to connect to database",
            "2024-01-01 10:00:01 ERROR Connection timeout",
            "2024-01-01 10:00:02 INFO Retrying connection",
            "2024-01-01 10:00:03 INFO Connected successfully",
        ]

        patterns = await analyzer.extract_patterns(log_lines)

        assert any("database" in p["pattern"] for p in patterns)
        assert any(p["level"] == "ERROR" for p in patterns)

    @pytest.mark.asyncio
    async def test_anomaly_detection_in_logs(self):
        """Test anomaly detection in logs."""
        from app.services.logging_analysis import LogAnalyzer

        analyzer = LogAnalyzer()

        # Normal logs followed by anomaly
        logs = []
        for i in range(100):
            logs.append(
                {
                    "timestamp": datetime.now() - timedelta(minutes=100 - i),
                    "level": "INFO",
                    "message": "Normal operation",
                }
            )

        # Add anomalies
        for i in range(10):
            logs.append(
                {
                    "timestamp": datetime.now() - timedelta(seconds=10 - i),
                    "level": "ERROR",
                    "message": "Critical error!",
                }
            )

        anomalies = await analyzer.detect_anomalies(logs)

        assert len(anomalies) > 0
        assert anomalies[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_log_aggregation_and_summary(self):
        """Test log aggregation and summary generation."""
        from app.services.logging_analysis import LogAnalyzer

        analyzer = LogAnalyzer()

        logs = []
        # Generate sample logs
        for i in range(1000):
            logs.append(
                {
                    "timestamp": datetime.now() - timedelta(minutes=1000 - i),
                    "level": ["INFO", "WARNING", "ERROR"][i % 3],
                    "component": ["api", "database", "cache"][i % 3],
                    "message": f"Log message {i}",
                }
            )

        summary = await analyzer.generate_summary(
            logs, group_by=["level", "component"], time_window="1h"
        )

        assert "total_logs" in summary
        assert "by_level" in summary
        assert "by_component" in summary
        assert summary["total_logs"] == 1000

    @pytest.mark.asyncio
    async def test_export_log_analysis_report(self):
        """Test exporting log analysis report."""
        from app.services.logging_analysis import LogAnalyzer

        analyzer = LogAnalyzer()

        analysis_results = {
            "summary": {"total_logs": 1000, "errors": 50},
            "patterns": [{"pattern": "database error", "count": 25}],
            "anomalies": [{"timestamp": "2024-01-01", "severity": "high"}],
        }

        with patch("builtins.open", mock_open()) as mock_file:
            await analyzer.export_report(
                analysis_results,
                format="json",
                output_path="/tmp/test_log_analysis.json",  # noqa: S108
            )

            mock_file.assert_called_once()
