"""
Sanjivani AI - Twitter Streamer

Stream and process tweets from Twitter/X API for flood crisis monitoring.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

FLOOD_KEYWORDS = [
    "flood", "flooding", "submerged", "waterlogged", "rescue",
    "help", "emergency", "stranded", "baadh", "bachao",
    "bihar", "patna", "darbhanga", "muzaffarpur", "kosi",
]


class TwitterStreamer:
    """Twitter API client for streaming flood-related tweets."""
    
    def __init__(self):
        self.bearer_token = settings.twitter_bearer_token
        self._client = None
    
    @property
    def is_configured(self) -> bool:
        return self.bearer_token is not None
    
    def search_recent(self, query: Optional[str] = None, max_results: int = 100) -> List[Dict]:
        """Search recent tweets. Returns empty list if not configured."""
        if not self.is_configured:
            logger.warning("Twitter not configured")
            return []
        
        try:
            import tweepy
            client = tweepy.Client(bearer_token=self.bearer_token)
            query = query or " OR ".join(FLOOD_KEYWORDS[:5])
            tweets = client.search_recent_tweets(query=query, max_results=min(max_results, 100))
            return [{"id": str(t.id), "text": t.text} for t in (tweets.data or [])]
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return []


class MockTwitterStreamer:
    """Mock streamer for development without Twitter API."""
    
    is_configured = True
    _samples = [
        "URGENT: Flood in Darbhanga! Need rescue! #BiharFloods",
        "Family stranded in Saharsa. 5 people including children. SOS!",
        "Relief camp at Patna University. Food available. #FloodRelief",
        "River Kosi overflowing near Supaul. Evacuate now!",
        "Need medical help in Khagaria. Elderly stuck without medicines.",
    ]
    
    def search_recent(self, query: Optional[str] = None, max_results: int = 10) -> List[Dict]:
        import random
        return [{"id": f"mock_{i}", "text": t, "created_at": datetime.now().isoformat()}
                for i, t in enumerate(random.sample(self._samples, min(max_results, len(self._samples))))]
    
    async def stream(self, callback: Optional[Callable] = None, max_tweets: int = 10):
        for i, tweet in enumerate(self.search_recent(max_results=max_tweets)):
            if callback: callback(tweet)
            yield tweet
            await asyncio.sleep(1)


def get_twitter_streamer(use_mock: bool = False):
    """Get streamer instance. Uses mock if Twitter not configured."""
    if use_mock or not settings.twitter_bearer_token:
        return MockTwitterStreamer()
    return TwitterStreamer()
