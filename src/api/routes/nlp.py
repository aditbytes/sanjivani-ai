"""
Sanjivani AI - NLP API Endpoints
"""

from typing import List

from fastapi import APIRouter, HTTPException

from src.api.schemas.tweet import TweetInput, TweetAnalysisResponse, BatchTweetInput
from src.nlp.pipeline import get_nlp_pipeline
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/analyze-tweet", response_model=TweetAnalysisResponse)
async def analyze_tweet(tweet: TweetInput) -> TweetAnalysisResponse:
    """Analyze a crisis tweet for urgency, resources, and location."""
    try:
        pipeline = get_nlp_pipeline()
        result = pipeline.analyze(tweet.text)
        
        return TweetAnalysisResponse(
            text=tweet.text,
            processed_text=result.get("processed_text", ""),
            urgency=result.get("urgency", "Medium"),
            urgency_confidence=result.get("urgency_confidence", 0.0),
            resource_needed=result.get("resource_needed"),
            vulnerability=result.get("vulnerability"),
            district=result.get("district"),
            latitude=result.get("latitude"),
            longitude=result.get("longitude"),
            inference_time_ms=result.get("inference_time_ms", 0.0),
        )
    except Exception as e:
        logger.error(f"Tweet analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-tweets")
async def analyze_tweets_batch(batch: BatchTweetInput) -> List[TweetAnalysisResponse]:
    """Analyze multiple tweets."""
    pipeline = get_nlp_pipeline()
    results = []
    
    for text in batch.texts:
        try:
            result = pipeline.analyze(text)
            results.append(TweetAnalysisResponse(
                text=text,
                processed_text=result.get("processed_text", ""),
                urgency=result.get("urgency", "Medium"),
                urgency_confidence=result.get("urgency_confidence", 0.0),
                resource_needed=result.get("resource_needed"),
                vulnerability=result.get("vulnerability"),
                district=result.get("district"),
                latitude=result.get("latitude"),
                longitude=result.get("longitude"),
                inference_time_ms=result.get("inference_time_ms", 0.0),
            ))
        except Exception as e:
            logger.error(f"Batch analysis error: {e}")
    
    return results
