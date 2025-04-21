# 🔭 Scout: AI-Powered Slack Assistant (Proof of Concept)
Scout is a lightweight RAG (Retrieval-Augmented Generation) assistant built to demonstrate how natural language interfaces can simplify querying business data. Designed to live in Slack, Scout connects to both SQLite and BigQuery databases, using LangChain to interpret queries and Flask + ngrok to serve responses locally.

## 🚀 Features
🔍 Natural language -> SQL -> Interpretation and answers via LangChain agents

🧠 Retrieval-Augmented Generation for accurate, context-aware answers

🗂️ Supports both SQLite (for testing) and BigQuery (for scale)

🤝 Slack integration for real-time interaction

🌐 Local deployment with ngrok for quick testing and demos

## 🛠️ Tech Stack
🐍 Python

⚗️ Flask (lightweight API server)

🔗 LangChain (LLM orchestration + SQL agent)

🗂️ SQLite3 and BigQuery (data sources)

🤖 Slack SDK (bot integration)

📡 ngrok (for exposing local server to Slack)

## ⚙️ Setup
### Clone this repo

```bash
git clone https://github.com/yourusername/scout-slack-bot.git
cd scout-slack-bot
```
### Install dependencies

```bash
poetry install
```
**You'll need:**

- OpenAI API key

- Slack Bot token + Signing secret

- BigQuery credentials JSON

- Optional: ngrok auth token

### Create a .env file with:

```ini
OPENAI_API_KEY=your_key
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

### Start the server

```bash
python app.py
```
Expose with ngrok

```bash
ngrok http 5000
```

## 💬 Usage
Once added to a Slack workspace, Scout listens for mentions and responds to natural language questions like:

"@Scout What were sales last quarter?"
"@Scout How many active users in March?"
"@Scout Get me the top 5 products by revenue."

It will translate these to SQL queries, execute them against the relevant data source, and return a human-readable answer.

## 📌 Notes
This is a proof-of-concept — not production ready.

Security, error handling, and scaling have been kept minimal to focus on functionality.

Easily extendable to other databases, APIs, or knowledge sources.

## 🧭 Roadmap Ideas
✅ Add SQL generation agents

✅ Add BigQuery + SQLite switching

🔜 Add Vector Database support 

🔜 Add query saving and usage metrics for continous development 

🔜 Basic conversation history memory

🔜 Deploy on cloud + custom domain
