# 📁 agentic_social_listening
# Streamlit-based Agentic Social Listening Dashboard (RSS/News-Based Version + Optional LLM Strategy Agent)

import streamlit as st
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from collections import defaultdict
import datetime
import time
from transformers import pipeline
from keybert import KeyBERT
import feedparser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Agentic Social Listening Advisor", layout="wide")

# Title
st.title("📣 Agentic Social Listening Advisor")
st.markdown("Enter a product or brand name to analyze recent news articles, extract sentiment and themes, and generate strategic recommendations.")

# User Input
product = st.text_input("🔍 Product/Brand Name", placeholder="e.g., Apple Vision Pro")

# Trigger button
run_analysis = st.button("Run Analysis")

if run_analysis and product:
    st.markdown("---")

    timeline_log = []  # Timeline for agent step-by-step recording
    output_records = []  # Persisted output records

    # Step 1: Listener Agent using RSS feeds
    with st.expander("Step 1: Listener Agent - Collect Mentions from News Feeds"):
        st.write("🛠️ Collecting recent mentions from RSS news feeds...")
        feed_urls = [
            "http://feeds.bbci.co.uk/news/rss.xml",
            "http://rss.cnn.com/rss/edition.rss",
            "https://www.theverge.com/rss/index.xml"
        ]
        mentions = []
        for url in feed_urls:
            st.subheader(f"📰 Raw Feed Preview for {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # Show a few entries for debug
                st.write("🔹 Title:", entry.title)
                st.write("🔹 Summary:", entry.get("summary", ""))
                st.write("🔹 Link:", entry.link)
                st.markdown("---")

            for entry in feed.entries:
                if product.lower() in entry.title.lower() or product.lower() in entry.get("summary", "").lower():
                    mention_text = f"{entry.title}. {entry.get('summary', '')}"
                    mentions.append(mention_text)

        if not mentions:
            st.warning("No news mentions found. Using fallback demo data.")
            mentions = [
                f"Just tried the new {product} — absolutely mind-blowing! 😍",
                f"Not impressed by {product}, expected more for the price. 😕",
                f"{product} is a game changer for AR. Hats off to the team!",
                f"I returned my {product}. It felt too heavy and clunky.",
                f"{product} has the best display I’ve ever seen!"
            ]

        for m in mentions:
            st.write(f"• {m}")
        timeline_log.append({"step": "Listener Agent", "timestamp": str(datetime.datetime.now()), "status": f"Fetched {len(mentions)} news mentions."})

    # Step 2: Sentiment & Theme Extraction Agent (Open Source)
    with st.expander("Step 2: Theme & Sentiment Agent (Open Source)"):
        st.write("📊 Analyzing sentiment and extracting themes using HuggingFace + KeyBERT...")

        sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        keyword_extractor = KeyBERT()

        sentiment_counts = defaultdict(int)
        themes = defaultdict(lambda: {"count": 0})

        for mention in mentions:
            result = sentiment_model(mention)[0]
            label = result['label'].lower()
            sentiment_counts[label] += 1

            keywords = keyword_extractor.extract_keywords(mention, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
            for kw, _ in keywords:
                themes[kw]["count"] += 1

        # Visualization
        labels = [k.capitalize() for k in sentiment_counts.keys()]
        sizes = [v for v in sentiment_counts.values()]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sizes)]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

        st.markdown("**Top Themes:**")
        for theme, meta in sorted(themes.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
            st.write(f"• {theme} ({meta['count']} mentions)")

        timeline_log.append({"step": "Sentiment+Theme Agent", "timestamp": str(datetime.datetime.now()), "status": f"Analyzed {len(mentions)} posts."})

    # Step 3: Strategy Agent (Optional LLM or Rule-Based)
    with st.expander("Step 3: Strategy Agent (Rule-Based + Optional LLM)"):
        st.write("🧐 Generating recommendation using rules and optionally enhancing with LLM...")

        total_mentions = sum(sentiment_counts.values())
        positive_ratio = sentiment_counts['positive'] / total_mentions if total_mentions else 0
        negative_ratio = sentiment_counts['negative'] / total_mentions if total_mentions else 0

        if negative_ratio > 0.5:
            recommendation = f"Address the major concerns around {product}. Consider public Q&A or an apology campaign."
        elif positive_ratio > 0.7:
            recommendation = f"Capitalize on the positive buzz around {product}. Launch a referral or testimonial campaign."
        else:
            recommendation = f"Monitor {product} feedback closely. Consider targeted surveys to better understand user sentiment."

        top_theme = max(themes.items(), key=lambda x: x[1]["count"])[0] if themes else "product feedback"
        tweet = f"Thanks for your thoughts on {product}! We're exploring ways to improve {top_theme} based on your feedback."

        # Optional LLM-enhanced version
        if st.checkbox("Use open-source LLM to rewrite tweet?"):
            with st.spinner("🤖 Enhancing tweet using TinyLlama..."):
                tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                prompt = f"Rewrite this as a compelling brand tweet: {tweet}"
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=50)
                tweet = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")

        st.markdown("**Recommended Action:**")
        st.success(recommendation)
        st.markdown("**Suggested Tweet:**")
        st.code(tweet)

        strategy = {"recommendation": recommendation, "tweet": tweet}
        timeline_log.append({"step": "Strategy Agent", "timestamp": str(datetime.datetime.now()), "status": "Generated rule-based action plan and tweet (LLM-enhanced optional)."})

    # Step 4: User Agent (Feedback)
    with st.expander("Step 4: Your Feedback (User Agent)"):
        feedback = st.radio("Do you agree with the recommendation?", ["Yes", "No - Revise"], index=0)
        if 'feedback_log' not in st.session_state:
            st.session_state['feedback_log'] = []

        st.session_state['feedback_log'].append({
            'product': product,
            'feedback': feedback
        })

        if feedback == "No - Revise":
            revised_input = st.text_area("What would you like the strategy to focus on instead?")
            if revised_input:
                st.session_state['feedback_log'][-1]['revision'] = revised_input
                st.warning("Strategy agent would re-run with your revised focus.")
                timeline_log.append({"step": "User Agent Revision", "timestamp": str(datetime.datetime.now()), "status": "Received revision input from user."})
        else:
            timeline_log.append({"step": "User Agent Feedback", "timestamp": str(datetime.datetime.now()), "status": f"User agreed with recommendation."})

        st.markdown("**Session Feedback Log:**")
        st.json(st.session_state['feedback_log'])

    # Persist timeline + strategy output to DataFrame
    output_records.append({
        'product': product,
        'sentiment': dict(sentiment_counts),
        'themes': list(themes.keys()),
        'recommendation': strategy['recommendation'],
        'timestamp': str(datetime.datetime.now())
    })

    df = pd.DataFrame(output_records)
    with st.expander("📁 Download Results as CSV"):
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name="agentic_report.csv")

    # Timeline summary
    with st.expander("🤖 Agent Activity Timeline"):
        st.markdown("Chronological log of each agent's action")
        st.json(timeline_log)

    st.markdown("---")
    st.markdown("© 2025 Agentic AI Demo")

else:
    st.info("Enter a product name and click 'Run Analysis' to get started.")
