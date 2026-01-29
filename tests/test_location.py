"""
Tests for location extraction module.
"""

import pytest

from src.nlp.location_extractor import LocationExtractor, extract_location


class TestLocationExtractor:
    """Test location extraction functionality."""
    
    def test_district_extraction(self):
        text = "Flooding in Patna district, need help"
        extractor = LocationExtractor(use_spacy=False)
        locations = extractor.extract(text)
        assert len(locations) > 0
        assert any(loc["name"] == "Patna" for loc in locations)
    
    def test_multiple_districts(self):
        text = "Water rising in Patna and Darbhanga"
        extractor = LocationExtractor(use_spacy=False)
        locations = extractor.extract(text)
        district_names = [loc["name"] for loc in locations]
        assert "Patna" in district_names
        assert "Darbhanga" in district_names
    
    def test_alias_extraction(self):
        text = "Flooding in Motihari area"
        extractor = LocationExtractor(use_spacy=False)
        locations = extractor.extract(text)
        assert any(loc["name"] == "East Champaran" for loc in locations)
    
    def test_coordinates_returned(self):
        text = "Help needed in Muzaffarpur"
        extractor = LocationExtractor(use_spacy=False)
        locations = extractor.extract(text)
        assert len(locations) > 0
        assert "lat" in locations[0]
        assert "lon" in locations[0]
    
    def test_primary_extraction(self):
        text = "Flooding in Patna"
        extractor = LocationExtractor(use_spacy=False)
        primary = extractor.extract_primary(text)
        assert primary is not None
        assert primary["name"] == "Patna"
    
    def test_convenience_function(self):
        text = "Help in Gaya district"
        locations = extract_location(text)
        assert "Gaya" in locations


class TestNoLocation:
    """Test cases with no location."""
    
    def test_no_location_text(self):
        text = "Please send help urgently"
        extractor = LocationExtractor(use_spacy=False)
        locations = extractor.extract(text)
        assert len(locations) == 0
    
    def test_no_primary_location(self):
        text = "General emergency alert"
        extractor = LocationExtractor(use_spacy=False)
        primary = extractor.extract_primary(text)
        assert primary is None
