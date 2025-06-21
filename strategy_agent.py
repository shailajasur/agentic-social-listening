# strategy_agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 🔮 Strategy Agent
# Agentic traits: reasons independently, generates domain-specific advice from upstream agent outputs

def generate_strategy_llm(sentiment_counts, themes, product_name):
    """
    This agent:
    - Reads structured sentiment + theme data
    - Crafts a business recommendation and tweet
    - Uses a local open-source LLM (gpt2)
    """

    model_name = "gpt2"  # Using gpt2 for better coherence vs tiny models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Compile data into a prompt
    sentiment_summary = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()])
    top_themes = ", ".join(sorted(themes.keys(), key=lambda k: themes[k]['count'], reverse=True)[:3])

    prompt = (
        f"Product: {product_name}\n"
        f"Sentiment Summary: {sentiment_summary}\n"
        f"Top Themes: {top_themes}\n\n"
        f"Write a brief business recommendation and a sample social media tweet.\n"
        f"Format:\n"
        f"Recommendation: <text>\n"
        f"Tweet: <text>"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to parse
    recommendation = "Focus on addressing top themes in upcoming releases."
    tweet = "Thanks for the feedback! We're making improvements based on your input."

    if "Recommendation:" in output_text and "Tweet:" in output_text:
        try:
            recommendation = output_text.split("Recommendation:")[1].split("Tweet:")[0].strip()
            tweet = output_text.split("Tweet:")[1].strip()
        except Exception:
            pass  # fallback to default strings

    return {
        "recommendation": recommendation,
        "tweet": tweet
    }
