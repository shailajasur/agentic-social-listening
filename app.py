# 📁 agentic_social_listening
# Streamlit-based Agentic Social Listening Dashboard (Reddit-Based Version)

import streamlit as st
import matplotlib.pyplot as plt
from strategy_agent import generate_strategy
import os
import pandas as pd
import re
from collections import defaultdict
import datetime
import time
from transformers import pipeline
from keybert import KeyBERT
import requests

st.set_page_config(page_title="Agentic Social Listening Advisor", layout="wide")

# Title
st.title("📣 Agentic Social Listening Advisor")
st.markdown("Enter a product or brand name to analyze recent Reddit posts, extract sentiment and themes, and generate strategic recommendations.")

# User Input
product = st.text_input("🔍 Product/Brand Name", placeholder="e.g., Apple Vision Pro")

# Trigger button
run_analysis = st.button("Run Analysis")

if run_analysis and product:
    st.markdown("---")

    timeline_log = []  # Timeline for agent step-by-step recording
    output_records = []  # Persisted output records

    # Step 1: Listener Agent using Reddit Pushshift API
    with st.expander("Step 1: Listener Agent - Collect Mentions from Reddit"):
        st.write("🛠️ Collecting recent mentions from Reddit using Pushshift API...")
        url = f"https://api.pushshift.io/reddit/search/comment/?q={product}&size=10&sort=desc"
        response = requests.get(url)
        mentions = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get("data", []):
                if 'body' in item:
                    mentions.append(item['body'])

        if not mentions:
            st.warning("No Reddit mentions found. Using fallback demo data.")
            mentions = [
                f"Just tried the new {product} — absolutely mind-blowing! 😍",
                f"Not impressed by {product}, expected more for the price. 😕",
                f"{product} is a game changer for AR. Hats off to the team!",
                f"I returned my {product}. It felt too heavy and clunky.",
                f"{product} has the best display I’ve ever seen!"
            ]

        for m in mentions:
            st.write(f"• {m}")
        timeline_log.append({"step": "Listener Agent", "timestamp": str(datetime.datetime.now()), "status": f"Fetched {len(mentions)} Reddit mentions."})

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

    # Step 3: Strategy Agent
    with st.expander("Step 3: Strategy Agent"):
        st.write("🤔 Generating recommendation based on sentiment and themes...")
        strategy = generate_strategy(sentiment_counts, themes)
        st.markdown("**Recommended Action:**")
        st.success(strategy['recommendation'])
        st.markdown("**Suggested Tweet:**")
        st.code(strategy['tweet'])
        timeline_log.append({"step": "Strategy Agent", "timestamp": str(datetime.datetime.now()), "status": "Generated action plan and draft tweet."})

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
