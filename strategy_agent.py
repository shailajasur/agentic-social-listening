# strategy_agent.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🧠 Strategy Agent - Agentic because it reasons over prior insights
# and generates domain-specific, actionable business guidance without manual prompting

def generate_strategy_llm(sentiment_counts, themes, product_name):
    """
    This agent:
    - Interprets sentiment analysis and key themes
    - Generates a strategic recommendation + marketing tweet
    - Uses a small local language model to simulate autonomy
    """

    model_name = "sshleifer/tiny-gpt2"  # ✅ Lightweight model for Streamlit compatibility
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Format input into a structured prompt
    sentiment_summary = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()])
    top_themes = ", ".join(sorted(themes.keys(), key=lambda k: themes[k]['count'], reverse=True)[:3])

    prompt = (
        f"You are a strategic AI agent helping a company understand feedback.\n\n"
        f"Product: {product_name}\n"
        f"Sentiment Summary: {sentiment_summary}\n"
        f"Top Themes: {top_themes}\n\n"
        f"Based on this data, provide:\n"
        f"1. A short business recommendation\n"
        f"2. A sample tweet the product team could post\n\n"
        f"Output Format:\n"
        f"Recommendation: <text>\n"
        f"Tweet: <text>\n"
    )

    # Generate output
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse response
    recommendation = "Review customer feedback for further optimization."
    tweet = "Thanks for sharing your experience with us!"

    if "Recommendation:" in generated and "Tweet:" in generated:
        try:
            recommendation = generated.split("Recommendation:")[1].split("Tweet:")[0].strip()
            tweet = generated.split("Tweet:")[1].strip()
        except:
            pass  # fallback to defaults

    return {
        "recommendation": recommendation,
        "tweet": tweet
    }
