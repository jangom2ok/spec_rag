"""Simple test for search_diversity.py to achieve coverage."""

import os
from datetime import datetime, timedelta

os.environ["TESTING"] = "true"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.services.search_diversity import (  # noqa: E402
    ClusteringDiversifier,
    ClusterResult,
    DiversificationAlgorithm,
    DiversificationRequest,
    DiversificationResult,
    DiversityCandidate,
    DiversityConfig,
    MMRDiversifier,
    SearchDiversityService,
    TemporalDiversifier,
    TopicDiversifier,
)


class TestSearchDiversity:
    """Test search diversity functionality."""

    def test_imports(self):
        """Test that imports work."""
        # This ensures the module loads
        assert DiversificationAlgorithm is not None
        assert DiversityConfig is not None
        assert SearchDiversityService is not None

    def test_enum_values(self):
        """Test enum values."""
        assert DiversificationAlgorithm.MMR.value == "mmr"
        assert DiversificationAlgorithm.CLUSTERING.value == "clustering"
        assert DiversificationAlgorithm.TOPIC_BASED.value == "topic_based"
        assert DiversificationAlgorithm.TEMPORAL.value == "temporal"
        assert DiversificationAlgorithm.HYBRID.value == "hybrid"

    def test_diversity_config_creation(self):
        """Test DiversityConfig creation."""
        # Default config
        config = DiversityConfig()
        assert config.diversity_factor == 0.5
        assert config.max_results == 10
        assert config.enable_clustering is True

        # Custom config
        config2 = DiversityConfig(diversity_factor=0.7, max_results=20, cluster_count=5)
        assert config2.diversity_factor == 0.7
        assert config2.max_results == 20
        assert config2.cluster_count == 5

    def test_diversity_config_validation(self):
        """Test DiversityConfig validation."""
        # Invalid diversity_factor
        with pytest.raises(
            ValueError, match="diversity_factor must be between 0 and 1"
        ):
            DiversityConfig(diversity_factor=-0.1)

        with pytest.raises(
            ValueError, match="diversity_factor must be between 0 and 1"
        ):
            DiversityConfig(diversity_factor=1.1)

        # Invalid max_results
        with pytest.raises(ValueError, match="max_results must be greater than 0"):
            DiversityConfig(max_results=0)

        # Invalid cluster_count
        with pytest.raises(ValueError, match="cluster_count must be greater than 0"):
            DiversityConfig(cluster_count=-1)

    def test_diversity_candidate(self):
        """Test DiversityCandidate creation."""
        # Basic candidate
        candidate = DiversityCandidate(
            id="doc1", content="Test content", title="Test Title", score=0.95
        )

        assert candidate.id == "doc1"
        assert candidate.content == "Test content"
        assert candidate.title == "Test Title"
        assert candidate.score == 0.95
        assert candidate.embedding is not None  # Auto-generated
        assert isinstance(candidate.metadata, dict)

        # Candidate with custom embedding
        custom_embedding = np.array([0.1, 0.2, 0.3])
        candidate2 = DiversityCandidate(
            id="doc2",
            content="Content 2",
            title="Title 2",
            score=0.85,
            embedding=custom_embedding,
            metadata={"category": "test", "author": "John"},
        )

        assert candidate2.embedding is not None and np.array_equal(
            candidate2.embedding, custom_embedding
        )
        assert candidate2.metadata["category"] == "test"
        assert candidate2.metadata["author"] == "John"

    def test_diversification_request(self):
        """Test DiversificationRequest."""
        candidates = [
            DiversityCandidate("1", "Content 1", "Title 1", 0.9),
            DiversityCandidate("2", "Content 2", "Title 2", 0.8),
        ]

        request = DiversificationRequest(query="test query", candidates=candidates)

        assert request.query == "test query"
        assert len(request.candidates) == 2
        assert request.max_results is None
        assert request.diversity_factor is None
        assert request.preserve_top_results == 0
        assert request.diversification_criteria == ["topic", "category"]

        # With custom values
        request2 = DiversificationRequest(
            query="custom query",
            candidates=candidates,
            max_results=5,
            diversity_factor=0.3,
            preserve_top_results=1,
            diversification_criteria=["author", "topic"],
        )

        assert request2.max_results == 5
        assert request2.diversity_factor == 0.3
        assert request2.preserve_top_results == 1
        assert "author" in request2.diversification_criteria

    def test_cluster_result(self):
        """Test ClusterResult."""
        candidates = [DiversityCandidate("1", "Content", "Title", 0.9)]
        centroid = np.array([0.5, 0.5])

        cluster = ClusterResult(
            cluster_id=0, candidates=candidates, centroid=centroid, coherence_score=0.8
        )

        assert cluster.cluster_id == 0
        assert len(cluster.candidates) == 1
        assert np.array_equal(cluster.centroid, centroid)
        assert cluster.coherence_score == 0.8

    def test_diversification_result(self):
        """Test DiversificationResult."""
        candidates = [DiversityCandidate("1", "Content", "Title", 0.9)]

        result = DiversificationResult(
            success=True,
            query="test",
            diversified_candidates=candidates,
            diversification_time=0.123,
            original_count=10,
        )

        assert result.success is True
        assert result.query == "test"
        assert len(result.diversified_candidates) == 1
        assert result.diversification_time == 0.123
        assert result.original_count == 10
        assert result.error_message is None
        assert result.cache_hit is False
        assert isinstance(result.diversity_metrics, dict)

        # Test get_summary method
        summary = result.get_summary()
        assert summary["success"] is True
        assert summary["query"] == "test"
        assert summary["original_count"] == 10
        assert summary["diversified_count"] == 1
        assert summary["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_search_diversity_service_basic(self):
        """Test SearchDiversityService basic functionality."""
        config = DiversityConfig()
        service = SearchDiversityService(config)

        # Create test candidates
        candidates = []
        for i in range(5):
            candidate = DiversityCandidate(
                id=f"doc{i}",
                content=f"Content for document {i}",
                title=f"Document {i}",
                score=0.9 - i * 0.1,
                metadata={
                    "category": f"cat{i % 2}",
                    "topics": [f"topic{i % 3}"],
                    "timestamp": datetime.now().isoformat(),
                },
            )
            candidates.append(candidate)

        # Test basic diversification
        request = DiversificationRequest(
            query="test query", candidates=candidates, max_results=3
        )

        result = await service.diversify(request)

        assert result.success is True
        assert len(result.diversified_candidates) <= 3
        assert result.original_count == 5
        assert result.diversification_time > 0

    @pytest.mark.asyncio
    async def test_search_diversity_service_algorithms(self):
        """Test different algorithms."""
        candidates = [
            DiversityCandidate(f"doc{i}", f"Content {i}", f"Title {i}", 0.9 - i * 0.1)
            for i in range(5)
        ]

        # Test each algorithm
        for algo in DiversificationAlgorithm:
            config = DiversityConfig(algorithm=algo)
            service = SearchDiversityService(config)

            request = DiversificationRequest(
                query=f"test {algo.value}", candidates=candidates, max_results=3
            )

            result = await service.diversify(request)
            assert result.success is True
            assert len(result.diversified_candidates) <= 3

    @pytest.mark.asyncio
    async def test_search_diversity_edge_cases(self):
        """Test edge cases."""
        config = DiversityConfig()
        service = SearchDiversityService(config)

        # Empty candidates
        request1 = DiversificationRequest(query="empty test", candidates=[])
        result1 = await service.diversify(request1)
        assert result1.success is True
        assert len(result1.diversified_candidates) == 0

        # Single candidate
        single_candidate = DiversityCandidate("1", "Content", "Title", 0.9)
        request2 = DiversificationRequest(
            query="single test", candidates=[single_candidate]
        )
        result2 = await service.diversify(request2)
        assert result2.success is True
        assert len(result2.diversified_candidates) == 1

    @pytest.mark.asyncio
    async def test_preserve_top_results(self):
        """Test preserving top results."""
        config = DiversityConfig()
        service = SearchDiversityService(config)

        # Create candidates with clear score ordering
        candidates = [
            DiversityCandidate(f"doc{i}", f"Content {i}", f"Title {i}", 1.0 - i * 0.1)
            for i in range(5)
        ]

        request = DiversificationRequest(
            query="preserve test",
            candidates=candidates,
            max_results=3,
            preserve_top_results=2,
        )

        result = await service.diversify(request)
        assert result.success is True
        # First 2 should be doc0 and doc1
        assert result.diversified_candidates[0].id == "doc0"
        assert result.diversified_candidates[1].id == "doc1"

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test caching functionality."""
        config = DiversityConfig(enable_caching=True)
        service = SearchDiversityService(config)

        candidates = [
            DiversityCandidate(f"doc{i}", f"Content {i}", f"Title {i}", 0.9)
            for i in range(3)
        ]

        request = DiversificationRequest(
            query="cache test", candidates=candidates, max_results=2
        )

        # First call
        result1 = await service.diversify(request)
        assert result1.cache_hit is False

        # Second call should hit cache
        result2 = await service.diversify(request)
        assert result2.cache_hit is True

    @pytest.mark.asyncio
    async def test_mmr_diversifier(self):
        """Test MMR diversifier specifically."""
        config = DiversityConfig(algorithm=DiversificationAlgorithm.MMR)
        diversifier = MMRDiversifier(config)

        # Create candidates with embeddings
        candidates = []
        for i in range(5):
            embedding = np.random.rand(128)
            candidate = DiversityCandidate(
                f"doc{i}",
                f"Content {i}",
                f"Title {i}",
                0.9 - i * 0.1,
                embedding=embedding,
            )
            candidates.append(candidate)

        selected = await diversifier.select(candidates, max_results=3)
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_clustering_diversifier(self):
        """Test clustering diversifier."""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.CLUSTERING, cluster_count=2
        )
        diversifier = ClusteringDiversifier(config)

        candidates = [
            DiversityCandidate(f"doc{i}", f"Content {i}", f"Title {i}", 0.9)
            for i in range(4)
        ]

        selected = await diversifier.select(candidates, max_results=2)
        assert len(selected) <= 2

    @pytest.mark.asyncio
    async def test_topic_diversifier(self):
        """Test topic diversifier."""
        config = DiversityConfig(algorithm=DiversificationAlgorithm.TOPIC_BASED)
        diversifier = TopicDiversifier(config)

        candidates = []
        for i in range(4):
            candidate = DiversityCandidate(
                f"doc{i}",
                f"Content {i}",
                f"Title {i}",
                0.9,
                metadata={"topics": [f"topic{i % 2}", f"topic{(i+1) % 2}"]},
            )
            candidates.append(candidate)

        selected = await diversifier.select(candidates, max_results=2)
        assert len(selected) <= 2

    @pytest.mark.asyncio
    async def test_temporal_diversifier(self):
        """Test temporal diversifier."""
        config = DiversityConfig(algorithm=DiversificationAlgorithm.TEMPORAL)
        diversifier = TemporalDiversifier(config)

        candidates = []
        for i in range(4):
            timestamp = (datetime.now() - timedelta(days=i * 7)).isoformat()
            candidate = DiversityCandidate(
                f"doc{i}",
                f"Content {i}",
                f"Title {i}",
                0.9,
                metadata={"timestamp": timestamp},
            )
            candidates.append(candidate)

        selected = await diversifier.select(candidates, max_results=2)
        assert len(selected) <= 2

    @pytest.mark.asyncio
    async def test_hybrid_algorithm(self):
        """Test hybrid algorithm."""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.HYBRID,
            enable_clustering=True,
            enable_topic_diversity=True,
        )
        service = SearchDiversityService(config)

        candidates = [
            DiversityCandidate(
                f"doc{i}",
                f"Content {i}",
                f"Title {i}",
                0.9,
                metadata={"topics": [f"topic{i % 2}"], "category": f"cat{i % 2}"},
            )
            for i in range(5)
        ]

        request = DiversificationRequest(
            query="hybrid test", candidates=candidates, max_results=3
        )

        result = await service.diversify(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        config = DiversityConfig()
        service = SearchDiversityService(config)

        # Create a candidate that might cause issues
        bad_candidate = DiversityCandidate(
            id="bad",
            content=None,  # type: ignore  # None content
            title="Title",
            score=None,  # type: ignore  # None score
        )

        request = DiversificationRequest(query="error test", candidates=[bad_candidate])

        # Should handle gracefully
        result = await service.diversify(request)
        # Either success with handling or failure with error message
        assert result is not None
        if not result.success:
            assert result.error_message is not None


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=app.services.search_diversity",
            "--cov-report=term-missing",
        ]
    )
