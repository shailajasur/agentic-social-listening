from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_strategy_llm(sentiment_counts, themes, product_name):
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    sentiment_summary = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()])
    top_themes = ", ".join(sorted(themes.keys(), key=lambda k: themes[k]['count'], reverse=True)[:5])

    prompt = (
        f"You are a marketing strategist.\n"
        f"Product: {product_name}\n"
        f"Sentiment breakdown: {sentiment_summary}\n"
        f"Key themes: {top_themes}\n\n"
        f"Give a strategic recommendation and a suggested tweet in response to this data."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Postprocess to split recommendation and tweet if structured properly
    recommendation = generated_text.split("Suggested tweet:")[0].strip()
    tweet = generated_text.split("Suggested tweet:")[-1].strip()

    return {
        "recommendation": recommendation,
        "tweet": tweet
    }
