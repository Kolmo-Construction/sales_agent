# Feedback UI — Admin Setup

One-time setup, then a single command to start.

---

## Prerequisites

All standard pipeline prerequisites must be running first:
- Ollama with `gemma2:9b` and `llama3.2` pulled
- Qdrant (local Docker or Qdrant Cloud)
- PostgreSQL with `POSTGRES_DSN` set (production database)

---

## One-time setup

```bash
# 1. Install dependencies (adds streamlit)
pip install -r requirements.txt

# 2. Create the feedback database
createdb sales_agent_feedback
psql sales_agent_feedback < db/schema.sql

# 3. Add to .env
FEEDBACK_POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent_feedback
```

---

## Start the UI

```bash
streamlit run feedback/app.py
# → opens at http://localhost:8501
```

To share with testers on the same network:
```bash
streamlit run feedback/app.py --server.address 0.0.0.0
# → testers access at http://<your-ip>:8501
```

---

## After a testing round

```bash
# See what happened
python scripts/analyze_feedback.py

# Review failures and add them to the eval suite
python scripts/promote_feedback.py

# Run evals to check if the new cases are caught
bash scripts/run_evals.sh

# Reset feedback DB for the next round (keeps production DB untouched)
dropdb sales_agent_feedback && createdb sales_agent_feedback
psql sales_agent_feedback < db/schema.sql
```

---

## Share with testers

Send testers `feedback/TESTER_GUIDE.md` and the URL. No accounts or passwords needed.
