"""
Tests for helper utilities.
"""

import pytest

from src.utils.helpers import (
    normalize_whitespace, truncate_text, BIHAR_DISTRICTS,
    get_district_coordinates, get_all_district_names, format_timestamp,
)


class TestTextHelpers:
    """Test text utility functions."""
    
    def test_normalize_whitespace(self):
        text = "  Hello   World  "
        result = normalize_whitespace(text)
        assert result == "Hello World"
    
    def test_truncate_text(self):
        text = "This is a very long text"
        result = truncate_text(text, 10)
        assert len(result) <= 13  # 10 + "..."
        assert result.endswith("...")
    
    def test_truncate_short_text(self):
        text = "Short"
        result = truncate_text(text, 10)
        assert result == "Short"


class TestDistrictHelpers:
    """Test Bihar district utilities."""
    
    def test_districts_loaded(self):
        assert len(BIHAR_DISTRICTS) > 0
        assert "Patna" in BIHAR_DISTRICTS
    
    def test_get_coordinates(self):
        coords = get_district_coordinates("Patna")
        assert coords is not None
        assert "lat" in coords
        assert "lon" in coords
        assert 24 < coords["lat"] < 28
        assert 83 < coords["lon"] < 89
    
    def test_get_coordinates_invalid(self):
        coords = get_district_coordinates("InvalidDistrict")
        assert coords is None
    
    def test_get_all_names(self):
        names = get_all_district_names()
        assert isinstance(names, list)
        assert "Patna" in names
        assert len(names) >= 38  # Bihar has 38 districts


class TestTimestampHelpers:
    """Test timestamp utilities."""
    
    def test_format_timestamp(self):
        from datetime import datetime
        dt = datetime(2024, 8, 15, 10, 30, 0)
        result = format_timestamp(dt)
        assert "2024" in result
