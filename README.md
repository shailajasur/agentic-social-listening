# 🧠 Agentic Social Listening Advisor

A real-time agentic AI dashboard that analyzes public sentiment about any product or brand and recommends strategic communication responses.

Built with:
- 🟩 Streamlit (UI)
- 🧠 Modular Python agents (simulated LLM behavior)
- 📊 Matplotlib (charts)

---

## 🚀 Features

- Accepts user input for a product or brand name
- Simulated Listener Agent fetches social media mentions
- Theme & Sentiment Agent classifies emotional tone and extracts hot topics
- Strategy Agent recommends a public-facing response (e.g., suggested tweet)
- User Agent provides feedback to refine recommendations
- Visualizes sentiment breakdown as a pie chart

---

## 🛠️ Setup (Windows / macOS)

```bash
# Clone repo or copy code
cd agentic_social_listening

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux

# Install dependencies
pip install streamlit matplotlib
