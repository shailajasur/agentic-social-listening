# 📁 agentic_social_listening
# Streamlit-based Agentic Social Listening Dashboard

import streamlit as st
import matplotlib.pyplot as plt
from listener_agent import get_mentions
from theme_agent import analyze_sentiment_and_themes
from strategy_agent import generate_strategy

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

    # Step 1: Listener Agent
    with st.expander("Step 1: Listener Agent - Collect Mentions"):
        st.write("🛠️ Collecting mentions from simulated data source...")
        mentions = get_mentions(product)
        for m in mentions:
            st.write(f"• {m}")

    # Step 2: Theme+Sentiment Agent
    with st.expander("Step 2: Theme & Sentiment Agent"):
        st.write("📊 Analyzing sentiment and extracting major themes...")
        analysis = analyze_sentiment_and_themes(mentions)

        labels = list(analysis['sentiment'].keys())
        sizes = list(analysis['sentiment'].values())
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

        st.markdown("**Detected Themes:**")
        for theme, details in analysis['themes'].items():
            st.write(f"- {theme} ({details['sentiment'].capitalize()})")

    # Step 3: Strategy Agent
    with st.expander("Step 3: Strategy Agent"):
        st.write("🧠 Generating recommendation based on sentiment and themes...")
        strategy = generate_strategy(analysis['sentiment'], analysis['themes'])

        st.markdown("**Recommended Action:**")
        st.success(strategy['recommendation'])

        st.markdown("**Suggested Tweet:**")
        st.code(strategy['tweet'])

    # Step 4: User Agent (Feedback)
    with st.expander("Step 4: Your Feedback (User Agent)"):
        feedback = st.radio("Do you agree with the recommendation?", ["Yes", "No - Revise"], index=0)
        if feedback == "No - Revise":
            revised_input = st.text_area("What would you like the strategy to focus on instead?")
            if revised_input:
                st.warning("Strategy agent would re-run with your revised focus.")

    st.markdown("---")
    st.markdown("© 2025 Agentic AI Demo")

else:
    st.info("Enter a product name and click 'Run Analysis' to get started.")
