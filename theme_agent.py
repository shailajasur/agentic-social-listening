# theme_agent.py

def analyze_sentiment_and_themes(mentions: list) -> dict:
    """
    Simulates sentiment analysis and theme detection.
    Replace this with LLM or ML-based logic in a real implementation.
    """
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    themes = {}

    for text in mentions:
        text_lower = text.lower()
        if "love" in text_lower or "beautiful" in text_lower:
            sentiments["positive"] += 1
            themes.setdefault("Design", {"sentiment": "positive", "count": 0})
            themes["Design"]["count"] += 1
        elif "sucks" in text_lower or "overpriced" in text_lower or "heating" in text_lower:
            sentiments["negative"] += 1
            themes.setdefault("Performance Issues", {"sentiment": "negative", "count": 0})
            themes["Performance Issues"]["count"] += 1
        elif "waiting" in text_lower or "shipping" in text_lower:
            sentiments["negative"] += 1
            themes.setdefault("Shipping Delays", {"sentiment": "negative", "count": 0})
            themes["Shipping Delays"]["count"] += 1
        else:
            sentiments["neutral"] += 1

    total = sum(sentiments.values())
    sentiment_percentages = {
        "positive": round(sentiments["positive"] / total * 100),
        "negative": round(sentiments["negative"] / total * 100),
        "neutral": round(sentiments["neutral"] / total * 100),
    }

    return {
        "sentiment": sentiment_percentages,
        "themes": themes
    }
