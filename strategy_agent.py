# strategy_agent.py

def generate_strategy(sentiment_counts, themes):
    total = sum(sentiment_counts.values())
    positive_ratio = sentiment_counts.get("positive", 0) / total if total > 0 else 0
    negative_ratio = sentiment_counts.get("negative", 0) / total if total > 0 else 0

    if negative_ratio > 0.4:
        recommendation = "Address key concerns raised by users and consider a public response campaign."
    elif positive_ratio > 0.5:
        recommendation = "Capitalize on the positive sentiment with a targeted promotion or success story."
    else:
        recommendation = "Engage with your community to boost awareness and monitor evolving feedback."

    top_theme = max(themes.items(), key=lambda x: x[1]["count"])[0] if themes else "engagement"
    tweet = f"Our team is listening! We're working to improve your experience around {top_theme.lower()} — stay tuned. 💡 #CustomerVoice"

    return {
        "recommendation": recommendation,
        "tweet": tweet
    }
