# 📁 agentic_social_listening
# Streamlit-based Agentic Social Listening Dashboard (GPT-Free Version)

import streamlit as st
import matplotlib.pyplot as plt
from strategy_agent import generate_strategy
import os
import subprocess
import pandas as pd
import re
from collections import defaultdict
import datetime
import time
from transformers import pipeline
from keybert import KeyBERT

st.set_page_config(page_title="Agentic Social Listening Advisor", layout="wide")

# Title
st.title("📣 Agentic Social Listening Advisor")
st.markdown("Enter a product or brand name to analyze real-time sentiment, themes, and generate strategic recommendations.")

# User Input
product = st.text_input("🔍 Product/Brand Name", placeholder="e.g., Apple Vision Pro")

# Trigger button
run_analysis = st.button("Run Analysis")

if run_analysis and product:
    st.markdown("---")

    timeline_log = []  # Timeline for agent step-by-step recording
    output_records = []  # Persisted output records

    # Step 1: Listener Agent using snscrape
    with st.expander("Step 1: Listener Agent - Collect Mentions from Twitter"):
        st.write("🛠️ Collecting mentions from Twitter using snscrape...")
        command = f"snscrape --max-results 10 twitter-search '{product}'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        raw_output = result.stdout
        mentions = []
        for line in raw_output.splitlines():
            if line.strip():
                mentions.append(line.strip())
        if not mentions:
            st.warning("No real-time mentions found. Try another term or check your snscrape setup.")
        else:
            for m in mentions:
                st.write(f"• {m}")
        timeline_log.append({"step": "Listener Agent", "timestamp": str(datetime.datetime.now()), "status": f"Fetched {len(mentions)} mentions."})

    # Step 2: Sentiment & Theme Extraction Agent (Open Source)
    with st.expander("Step 2: Theme & Sentiment Agent (Open Source)"):
        st.write("📊 Analyzing sentiment and extracting themes using HuggingFace + KeyBERT...")

        # Load sentiment model
        sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        keyword_extractor = KeyBERT()

        sentiment_counts = defaultdict(int)
        themes = defaultdict(int)

        for mention in mentions:
            result = sentiment_model(mention)[0]
            label = result['label'].lower()
            sentiment_counts[label] += 1

            keywords = keyword_extractor.extract_keywords(mention, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
            for kw, _ in keywords:
                themes[kw] += 1

        # Visualization
        labels = [k.capitalize() for k in sentiment_counts.keys()]
        sizes = [v for v in sentiment_counts.values()]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sizes)]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

        st.markdown("**Top Themes:**")
        for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"• {theme} ({count} mentions)")

        timeline_log.append({"step": "Sentiment+Theme Agent", "timestamp": str(datetime.datetime.now()), "status": f"Analyzed {len(mentions)} posts."})

    # Step 3: Strategy Agent
    with st.expander("Step 3: Strategy Agent"):
        st.write("🧐 Generating recommendation based on sentiment and themes...")
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
