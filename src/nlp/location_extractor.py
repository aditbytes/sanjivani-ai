"""
Sanjivani AI - Location Extractor

Extract location mentions from crisis text using spaCy NER and Bihar gazetteer.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.utils.helpers import BIHAR_DISTRICTS, get_district_coordinates
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocationExtractor:
    """Extract locations from crisis text using NER and gazetteer matching."""
    
    # District name variations and aliases
    DISTRICT_ALIASES = {
        "east champaran": "East Champaran", "west champaran": "West Champaran",
        "motihari": "East Champaran", "bettiah": "West Champaran",
        "ara": "Bhojpur", "chapra": "Saran", "hajipur": "Vaishali",
        "sasaram": "Rohtas", "biharsharif": "Nalanda", "bihar sharif": "Nalanda",
    }
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._nlp = None
        self._matcher = None
        
        # Build lowercase lookup
        self.district_lookup = {name.lower(): name for name in BIHAR_DISTRICTS}
        self.district_lookup.update({k.lower(): v for k, v in self.DISTRICT_ALIASES.items()})
    
    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded")
            except Exception as e:
                logger.warning(f"Could not load spaCy: {e}")
                self._nlp = False
        return self._nlp if self._nlp else None
    
    def extract(self, text: str) -> List[Dict]:
        """
        Extract locations from text.
        
        Returns:
            List of dicts with 'name', 'lat', 'lon', 'confidence'
        """
        locations = []
        text_lower = text.lower()
        
        # Gazetteer matching
        for district_lower, district_name in self.district_lookup.items():
            if district_lower in text_lower:
                coords = get_district_coordinates(district_name)
                if coords:
                    locations.append({
                        "name": district_name,
                        "lat": coords["lat"],
                        "lon": coords["lon"],
                        "confidence": 0.9,
                        "method": "gazetteer",
                    })
        
        # spaCy NER
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    ent_lower = ent.text.lower()
                    if ent_lower in self.district_lookup:
                        district = self.district_lookup[ent_lower]
                        if not any(l["name"] == district for l in locations):
                            coords = get_district_coordinates(district)
                            if coords:
                                locations.append({
                                    "name": district,
                                    "lat": coords["lat"],
                                    "lon": coords["lon"],
                                    "confidence": 0.8,
                                    "method": "ner",
                                })
        
        return locations
    
    def extract_primary(self, text: str) -> Optional[Dict]:
        """Extract the primary (first/most confident) location."""
        locations = self.extract(text)
        if locations:
            return max(locations, key=lambda x: x["confidence"])
        return None


def extract_location(text: str) -> List[str]:
    """Convenience function to extract location names."""
    extractor = LocationExtractor()
    return [loc["name"] for loc in extractor.extract(text)]
