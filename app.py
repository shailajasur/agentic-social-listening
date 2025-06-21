# 📁 agentic_social_listening
# Streamlit-based Agentic Social Listening Dashboard

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from strategy_agent import generate_strategy
import os
import subprocess
import pandas as pd
import re
from collections import defaultdict
import datetime
import time
import openai

st.set_page_config(page_title="Agentic Social Listening Advisor", layout="wide")

# Set OpenAI API key from environment variable or manual entry
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Title
st.title("📣 Agentic Social Listening Advisor")
st.markdown("Enter a product or brand name to analyze real-time sentiment, themes, and generate strategic recommendations.")

# User Input
product = st.text_input("🔍 Product/Brand Name", placeholder="e.g., Apple Vision Pro")

# Trigger button
run_analysis = st.button("Run Analysis")

# Retry wrapper for GPT call with usage logging
usage_log = []

def safe_chat_completion(client, model, messages, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            usage = response.usage if hasattr(response, 'usage') else {}
            usage_log.append({"timestamp": str(datetime.datetime.now()), "tokens": usage})
            return response
        except openai.RateLimitError:
            st.warning(f"Rate limit hit. Retrying ({attempt + 1}/{retries}) in {delay} seconds...")
            time.sleep(delay * (attempt + 1))
        except openai.OpenAIError as e:
            st.error(f"OpenAI API error: {str(e)}")
            break
    st.error("❌ GPT API failed after multiple retries.")
    return None

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
            if line.strip().startswith(product):
                mentions.append(line.strip())
        if not mentions:
            st.warning("No real-time mentions found. Try another term or check your snscrape setup.")
        else:
            for m in mentions:
                st.write(f"• {m}")
        timeline_log.append({"step": "Listener Agent", "timestamp": str(datetime.datetime.now()), "status": f"Fetched {len(mentions)} mentions."})

    # Step 2: Theme+Sentiment Agent with GPT-3.5
    with st.expander("Step 2: Theme & Sentiment Agent (GPT-3.5 Powered)"):
        st.write("📊 Analyzing sentiment and extracting major themes with GPT-3.5...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        response = safe_chat_completion(
            openai_client,
            "gpt-3.5-turbo",
            [
                {"role": "system", "content": "You are a social insights analyst. Classify sentiment (Positive, Negative, Neutral) for each post, and extract top 3 common themes."},
                {"role": "user", "content": f"Here are the mentions:\n" + "\n".join(mentions)}
            ]
        )
        if not response:
            st.stop()

        parsed = response.choices[0].message.content
        st.markdown("**GPT Analysis Output:**")
        st.text(parsed)

        sentiment_counts = defaultdict(int)
        themes = {}
        for line in parsed.splitlines():
            sentiment_match = re.search(r"Sentiment: (Positive|Negative|Neutral)", line)
            theme_match = re.findall(r"Theme: ([^\n,]+)", line)
            if sentiment_match:
                sentiment = sentiment_match.group(1).lower()
                sentiment_counts[sentiment] += 1
            for t in theme_match:
                themes[t.strip()] = {'sentiment': sentiment, 'count': themes.get(t.strip(), {}).get('count', 0) + 1}

        labels = [k.capitalize() for k in sentiment_counts.keys()]
        sizes = [v for v in sentiment_counts.values()]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sizes)]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)
        timeline_log.append({"step": "Sentiment Agent", "timestamp": str(datetime.datetime.now()), "status": f"Identified {len(sentiment_counts)} sentiment categories and {len(themes)} themes."})

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
    with st.expander("🤮 Agent Activity Timeline"):
        st.markdown("Chronological log of each agent's action")
        st.json(timeline_log)

    with st.expander("📊 GPT Token Usage Log"):
        st.json(usage_log)

    st.markdown("---")
    st.markdown("© 2025 Agentic AI Demo")

else:
    st.info("Enter a product name and click 'Run Analysis' to get started.")
