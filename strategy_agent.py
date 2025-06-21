# strategy_agent.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_strategy_llm(sentiment_counts, themes, product_name):
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    sentiment_summary = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()])
    top_themes = ", ".join(sorted(themes.keys(), key=lambda k: themes[k]['count'], reverse=True)[:3])

    prompt = (
        f"Product: {product_name}\n"
        f"Sentiment: {sentiment_summary}\n"
        f"Themes: {top_themes}\n"
        f"What should the company do? Provide a strategy and a tweet."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Fallback parsing logic
    recommendation = generated.split("\n")[0]
    tweet = generated.split("\n")[1] if len(generated.split("\n")) > 1 else "Thanks for your feedback!"

    return {
        "recommendation": recommendation,
        "tweet": tweet
    }
