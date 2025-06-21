# strategy_agent.py

def generate_strategy(sentiment: dict, themes: dict) -> dict:
    """
    Generates a recommendation based on sentiment distribution and dominant themes.
    In real-world use, you would use GPT or a rules engine.
    """
    dominant_negative = [
        theme for theme, meta in themes.items()
        if meta["sentiment"] == "negative" and meta["count"] > 1
    ]

    if not dominant_negative:
        recommendation = "Promote the positive themes in marketing campaigns."
        sugg
