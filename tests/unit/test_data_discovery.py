# Unit Tests - Data Discovery Module

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.discovery import DatasetDiscovery, DatasetInfo, DataMerger


# ==================== Test DatasetInfo ====================

class TestDatasetInfo:
    """Test DatasetInfo dataclass."""

    def test_dataset_info_creation(self):
        """DatasetInfo can be created with required fields."""
        ds = DatasetInfo(
            source="roboflow",
            name="test-dataset",
            url="https://example.com",
            license="MIT",
            annotations="coco",
            images=1000,
            categories=["car", "truck"],
        )

        assert ds.source == "roboflow"
        assert ds.name == "test-dataset"
        assert ds.images == 1000
        assert ds.relevance_score == 0.0  # Default

    def test_dataset_info_with_relevance(self):
        """DatasetInfo can be created with relevance score."""
        ds = DatasetInfo(
            source="kaggle",
            name="test",
            url="",
            license="",
            annotations="",
            images=0,
            categories=[],
            relevance_score=0.85,
        )

        assert ds.relevance_score == 0.85


# ==================== Test Relevance Scoring ====================

class TestRelevanceScoring:
    """Test _calculate_relevance method."""

    def test_exact_match_returns_1(self, discovery_instance):
        """Exact match should return score of 1.0."""
        score = discovery_instance._calculate_relevance("car", "car")
        assert score == 1.0

    def test_exact_match_case_insensitive(self, discovery_instance):
        """Case insensitive exact match."""
        score = discovery_instance._calculate_relevance("Car", "CAR")
        assert score == 1.0

    def test_query_in_text_returns_09(self, discovery_instance):
        """Query contained in text returns 0.9."""
        score = discovery_instance._calculate_relevance("car", "car_detection")
        assert score == 0.9

    def test_text_in_query_returns_08(self, discovery_instance):
        """Text contained in query returns 0.8."""
        score = discovery_instance._calculate_relevance("car_detection", "car")
        assert score == 0.8

    def test_empty_query_returns_0(self, discovery_instance):
        """Empty query returns 0.0."""
        score = discovery_instance._calculate_relevance("", "car")
        assert score == 0.0

    def test_empty_text_returns_0(self, discovery_instance):
        """Empty text returns 0.0."""
        score = discovery_instance._calculate_relevance("car", "")
        assert score == 0.0

    def test_both_empty_returns_0(self, discovery_instance):
        """Both empty returns 0.0."""
        score = discovery_instance._calculate_relevance("", "")
        assert score == 0.0

    def test_jaccard_similarity(self, discovery_instance):
        """Jaccard similarity for word-level matching."""
        score = discovery_instance._calculate_relevance("car vehicle", "car truck")
        # Should have some overlap but not exact
        assert 0.0 <= score <= 1.0

    def test_no_common_words(self, discovery_instance):
        """No common words returns low score."""
        score = discovery_instance._calculate_relevance("car", "building")
        assert score < 0.5

    def test_boundary_word_boost(self, discovery_instance):
        """Word at boundary gets boost."""
        # "car" at start of "car_detection" should get boost
        score = discovery_instance._calculate_relevance("car", "car_something")
        # Should be higher due to boundary match
        assert score > 0.8


# ==================== Test Dataset Discovery ====================

class TestDatasetDiscovery:
    """Test DatasetDiscovery class."""

    def test_initialization(self, temp_dir):
        """Discovery initializes with output directory."""
        discovery = DatasetDiscovery(output_dir=temp_dir)
        assert discovery.output_dir == temp_dir
        assert temp_dir.exists()

    def test_initialization_default_dir(self):
        """Discovery initializes with default directory."""
        discovery = DatasetDiscovery()
        assert discovery.output_dir.exists()

    def test_search_roboflow_no_api_key(self, discovery_instance):
        """Roboflow search returns empty without API key."""
        # Without API key, should return empty
        import os
        original_key = os.environ.get('ROBOFLOW_API_KEY')
        try:
            # Unset the env var
            if 'ROBOFLOW_API_KEY' in os.environ:
                del os.environ['ROBOFLOW_API_KEY']
            results = discovery_instance._search_roboflow("test", 5)
            assert results == []
        finally:
            # Restore
            if original_key:
                os.environ['ROBOFLOW_API_KEY'] = original_key

    def test_search_kaggle_handles_error(self, discovery_instance):
        """Kaggle error handling returns empty list."""
        # Test with Kaggle not available
        import data.discovery as dd
        original = getattr(dd, 'KAGGLE_AVAILABLE', True)
        dd.KAGGLE_AVAILABLE = False
        try:
            results = discovery_instance._search_kaggle("test", 5)
            assert results == []
        finally:
            dd.KAGGLE_AVAILABLE = original

    def test_search_huggingface_handles_error(self, discovery_instance):
        """HuggingFace error handling returns empty list when library not available."""
        import data.discovery as dd
        # When DATASETS_AVAILABLE is False, should return empty
        original = getattr(dd, 'DATASETS_AVAILABLE', False)
        dd.DATASETS_AVAILABLE = False
        try:
            results = discovery_instance._search_huggingface("test", 5)
            assert results == []
        finally:
            dd.DATASETS_AVAILABLE = original

    def test_search_calls_all_sources(self, discovery_instance):
        """Search queries all three sources."""
        with patch.object(discovery_instance, '_search_roboflow', return_value=[]) as mock_rf:
            with patch.object(discovery_instance, '_search_kaggle', return_value=[]) as mock_kg:
                with patch.object(discovery_instance, '_search_huggingface', return_value=[]) as mock_hf:
                    discovery_instance.search("test", max_results=10)

                    mock_rf.assert_called_once_with("test", 10)
                    mock_kg.assert_called_once_with("test", 10)
                    mock_hf.assert_called_once_with("test", 10)

    def test_search_sorts_by_relevance(self, discovery_instance):
        """Search results are sorted by relevance score."""
        with patch.object(discovery_instance, '_search_roboflow', return_value=[
            DatasetInfo("roboflow", "a", "", "", "", 0, [], relevance_score=0.3),
            DatasetInfo("roboflow", "b", "", "", "", 0, [], relevance_score=0.8),
            DatasetInfo("roboflow", "c", "", "", "", 0, [], relevance_score=0.5),
        ]):
            with patch.object(discovery_instance, '_search_kaggle', return_value=[]):
                with patch.object(discovery_instance, '_search_huggingface', return_value=[]):
                    results = discovery_instance.search("test")

                    # Should be sorted descending
                    assert results[0].relevance_score == 0.8
                    assert results[1].relevance_score == 0.5
                    assert results[2].relevance_score == 0.3


# ==================== Test Data Merger ====================

class TestDataMerger:
    """Test DataMerger class."""

    def test_initialization(self):
        """Merger initializes with ratio limit."""
        merger = DataMerger(max_synthetic_ratio=0.3)
        assert merger.max_synthetic_ratio == 0.3

    def test_initialization_default(self):
        """Merger has correct default ratio."""
        merger = DataMerger()
        assert merger.max_synthetic_ratio == 0.3

    def test_merge_empty_input(self, temp_dir, data_merger_instance):
        """Merge handles empty input gracefully."""
        result = data_merger_instance.merge(
            discovered_datasets=[],
            output_dir=temp_dir / "merged"
        )

        assert result["train_images"] == 0
        assert result["val_images"] == 0

    def test_count_images_empty_dir(self, data_merger_instance, temp_dir):
        """Count images handles empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        count = data_merger_instance._count_images(empty_dir)
        assert count == 0

    def test_count_images_with_files(self, data_merger_instance, temp_dir):
        """Count images correctly counts image files."""
        img_dir = temp_dir / "images"
        img_dir.mkdir()

        # Create test images
        (img_dir / "test1.jpg").touch()
        (img_dir / "test2.png").touch()
        (img_dir / "test3.jpeg").touch()
        (img_dir / "test4.txt").touch()  # Should not count

        count = data_merger_instance._count_images(img_dir)
        assert count == 3
