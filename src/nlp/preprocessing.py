"""
Sanjivani AI - NLP Text Preprocessing

Text cleaning and normalization for crisis tweets in English, Hindi, and Hinglish.
"""

import re
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Hinglish spelling normalization mappings
HINGLISH_MAPPINGS = {
    "plz": "please", "pls": "please", "hlp": "help", "halp": "help",
    "ppl": "people", "govt": "government", "govt.": "government",
    "bcoz": "because", "frnd": "friend", "frnds": "friends",
    "msg": "message", "msgs": "messages", "ur": "your", "u": "you",
    "r": "are", "2": "to", "4": "for", "b4": "before",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks",
    "bro": "brother", "sis": "sister", "asap": "as soon as possible",
    "nd": "and", "abt": "about", "wid": "with", "widout": "without",
    "bcz": "because", "coz": "because", "bt": "but", "nt": "not",
    "sry": "sorry", "srry": "sorry", "v": "we", "hv": "have",
    "bht": "bahut", "bhut": "bahut", "kya": "what", "kyu": "why",
    "kyun": "why", "kese": "how", "kaise": "how", "hai": "is",
    "hain": "are", "ho": "are", "mujhe": "me", "hamara": "our",
}


class TextPreprocessor:
    """Text preprocessing pipeline for crisis tweets."""
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtag_symbol: bool = True,
        normalize_hinglish: bool = True,
        lowercase: bool = True,
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtag_symbol = remove_hashtag_symbol
        self.normalize_hinglish = normalize_hinglish
        self.lowercase = lowercase
        
        # Compile regex patterns
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._mention_pattern = re.compile(r"@\w+")
        self._hashtag_pattern = re.compile(r"#(\w+)")
        self._whitespace_pattern = re.compile(r"\s+")
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )
    
    def __call__(self, text: str) -> str:
        """Process text through the full pipeline."""
        return self.preprocess(text)
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text."""
        if not text:
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub(" ", text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self._mention_pattern.sub(" ", text)
        
        # Process hashtags (keep word, remove #)
        if self.remove_hashtag_symbol:
            text = self._hashtag_pattern.sub(r"\1", text)
        
        # Normalize Hinglish spellings
        if self.normalize_hinglish:
            text = self._normalize_hinglish(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = self._whitespace_pattern.sub(" ", text)
        
        return text.strip()
    
    def _normalize_hinglish(self, text: str) -> str:
        """Normalize Hinglish spellings to standard form."""
        words = text.split()
        normalized = []
        for word in words:
            word_lower = word.lower()
            if word_lower in HINGLISH_MAPPINGS:
                normalized.append(HINGLISH_MAPPINGS[word_lower])
            else:
                normalized.append(word)
        return " ".join(normalized)


def preprocess_tweet(text: str) -> str:
    """Convenience function for preprocessing a single tweet."""
    preprocessor = TextPreprocessor()
    return preprocessor(text)


def batch_preprocess(texts: List[str]) -> List[str]:
    """Preprocess a batch of texts."""
    preprocessor = TextPreprocessor()
    return [preprocessor(text) for text in texts]
