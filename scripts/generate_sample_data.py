#!/usr/bin/env python
"""
Sanjivani AI - Sample Data Generator

Generate synthetic data for development and testing.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.helpers import BIHAR_DISTRICTS, get_district_coordinates

SAMPLE_TWEETS = [
    "URGENT! Water level rising rapidly in {district}. Need immediate rescue!",
    "Marooned on rooftop in {district} with elderly parents. Please send help!",
    "Flood water entered homes in {district}. Children need medical attention.",
    "No food for two days in {district}. Please send relief supplies.",
    "{district} completely submerged. Boats needed urgently! #BiharFlood",
    "My grandmother is sick in {district}. Water everywhere, can't reach hospital.",
    "School children trapped in {district}. Need immediate evacuation.",
    "Animals dying in {district}. Farmers need fodder and shelter.",
    "Electricity down for 48 hours in {district}. Pregnant woman needs hospital.",
    "Please help! Our village in {district} cut off. No road connectivity.",
]

URGENCY_LEVELS = ["Critical", "High", "Medium", "Low", "Non-Urgent"]
RESOURCES = ["Rescue", "Medical", "Food", "Shelter", "Information", "Water"]
VULNERABILITIES = ["Elderly", "Children", "Disabled", "Pregnant", "None"]


def generate_sample_tweets(n: int = 500) -> list:
    """Generate synthetic crisis tweets."""
    tweets = []
    districts = list(BIHAR_DISTRICTS.keys())
    
    for i in range(n):
        district = random.choice(districts)
        template = random.choice(SAMPLE_TWEETS)
        text = template.format(district=district)
        coords = get_district_coordinates(district)
        
        tweets.append({
            "id": f"tweet_{i+1}",
            "text": text,
            "district": district,
            "latitude": coords["lat"] + random.uniform(-0.1, 0.1) if coords else None,
            "longitude": coords["lon"] + random.uniform(-0.1, 0.1) if coords else None,
            "urgency": random.choices(URGENCY_LEVELS, weights=[10, 25, 35, 20, 10])[0],
            "resource_needed": random.choice(RESOURCES),
            "vulnerability": random.choice(VULNERABILITIES),
            "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 72))).isoformat(),
        })
    
    return tweets


def generate_historical_floods(n: int = 50) -> list:
    """Generate synthetic historical flood data."""
    events = []
    districts = list(BIHAR_DISTRICTS.keys())
    
    for i in range(n):
        events.append({
            "id": f"flood_{i+1}",
            "primary_district": random.choice(districts),
            "affected_districts": random.sample(districts, k=random.randint(1, 5)),
            "start_date": f"{random.randint(2018, 2023)}-{random.randint(6, 9):02d}-{random.randint(1, 28):02d}",
            "duration_days": random.randint(5, 30),
            "affected_population": random.randint(10000, 500000),
            "food_packets": random.randint(5000, 100000),
            "medical_kits": random.randint(500, 10000),
            "rescue_boats": random.randint(50, 500),
            "shelters_established": random.randint(10, 100),
        })
    
    return events


def main():
    """Generate and save sample data."""
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tweets
    tweets = generate_sample_tweets(500)
    with open(raw_dir / "sample_tweets.json", "w") as f:
        json.dump(tweets, f, indent=2)
    
    # Split into train/val/test
    random.shuffle(tweets)
    n = len(tweets)
    train, val, test = tweets[:int(n*0.7)], tweets[int(n*0.7):int(n*0.85)], tweets[int(n*0.85):]
    
    with open(processed_dir / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(processed_dir / "val.json", "w") as f:
        json.dump(val, f, indent=2)
    with open(processed_dir / "test.json", "w") as f:
        json.dump(test, f, indent=2)
    
    # Generate historical data
    floods = generate_historical_floods(50)
    with open(processed_dir / "historical_floods.json", "w") as f:
        json.dump(floods, f, indent=2)
    
    print(f"Generated {len(tweets)} tweets and {len(floods)} historical events")
    print(f"Data saved to {data_dir}")


if __name__ == "__main__":
    main()
