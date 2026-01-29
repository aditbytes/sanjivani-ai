"""
Tests for NLP preprocessing module.
"""

import pytest

from src.nlp.preprocessing import TextPreprocessor, preprocess_tweet


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_url_removal(self):
        text = "Help needed https://example.com/flood"
        processor = TextPreprocessor()
        result = processor(text)
        assert "https://" not in result
        assert "example.com" not in result
    
    def test_mention_removal(self):
        text = "@NDRF please help us in Patna"
        processor = TextPreprocessor()
        result = processor(text)
        assert "@NDRF" not in result
        assert "patna" in result
    
    def test_hashtag_processing(self):
        text = "#BiharFlood help needed"
        processor = TextPreprocessor()
        result = processor(text)
        assert "#" not in result
        assert "biharflood" in result
    
    def test_hinglish_normalization(self):
        text = "plz hlp us. govt shld send rescue"
        processor = TextPreprocessor()
        result = processor(text)
        assert "please" in result
        assert "help" in result
        assert "government" in result
    
    def test_lowercase(self):
        text = "URGENT HELP NEEDED"
        processor = TextPreprocessor()
        result = processor(text)
        assert result == "urgent help needed"
    
    def test_convenience_function(self):
        text = "Test tweet @user https://t.co/abc"
        result = preprocess_tweet(text)
        assert isinstance(result, str)
        assert "@user" not in result


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing."""
    
    def test_empty_string(self):
        result = preprocess_tweet("")
        assert result == ""
    
    def test_only_url(self):
        result = preprocess_tweet("https://example.com")
        assert result.strip() == ""
    
    def test_multiple_spaces(self):
        text = "Help    needed   in    Patna"
        result = preprocess_tweet(text)
        assert "  " not in result
    
    def test_hindi_text_preserved(self):
        text = "मदद चाहिए Patna में"
        result = preprocess_tweet(text)
        assert "मदद" in result
