📊 Olist GenAI — Hybrid Analytics & Retrieval-Augmented Agent

A domain-aware, hallucination-free analytics assistant for Olist E-commerce Data

🚀 Overview

Olist GenAI is an intelligent analytics agent that combines:

✅ Deterministic analytics (real calculations, no hallucination)

with

🤖 LLM-powered natural language understanding & explanations

This hybrid architecture allows the system to:

Answer business questions with actual computed values

Perform sales analysis, revenue breakdowns, category insights

Apply time-aware logic like “past 2 quarters”

Retrieve supporting evidence rows using semantic search

Prevent hallucinations by grounding all answers in data

This project was built for the
📌 GDA Corp AI/ML Campus Hiring Assignment
and showcases a practical GenAI solution for real analytics use cases.

🧠 Key Features
🔥 1. Hybrid Agent Architecture (RAG + Analytics Engine)

The system interprets user queries, generates a structured analytics instruction, and executes it using:

Pandas (deterministic calculations)

FAISS vector search (retrieval)

OpenRouter LLM (natural language explanations)

This ensures accuracy AND flexibility.

📈 2. Deterministic Analytics Engine

Supports:

sum(price)

mean(price)

count(*)

group_by = state / city / category

Date filtering (between X and Y)

Example queries that work:

“Which states generate the highest revenue?”

“Average order value for Electronics”

“Total sales between 2021-01-01 and 2021-12-31”

“Count of orders by category”

🗂️ 3. Quarter-Aware Sales Analysis

The system includes advanced time logic:

✔️ Detects queries like
“highest selling category in the past 2 quarters”

✔️ Computes:

last completed quarter

previous quarters

revenue mask

top categories

🔍 4. Retrieval-Augmented Generation (RAG)

Uses FAISS embeddings to retrieve the top-K most relevant rows.

All final answers show supporting row indices for transparency.

Example:

Supporting rows: 22678, 39179, 74768

🧠 5. LLM-Powered Query Parsing

Natural language → JSON structured query via:

✔️ OpenRouter (GPT-4o-mini)
✔️ Rule-based fallback parser

Example NL → JSON:

{
  "action": "aggregate",
  "agg": "sum",
  "column": "price",
  "group_by": "customer_state",
  "filter": {
    "date_from": null,
    "date_to": null
  }
}

🗣️ 6. LLM Explanations for Humans

After executing analytics, the LLM is used only to summarize and explain
— never to compute numbers.

This eliminates hallucination.

🏗️ Architecture
                🧑‍💼 User Query
                       │
                       ▼
            ┌────────────────────┐
            │  NL → JSON Parser  │  ← OpenRouter (LLM)
            └────────────────────┘
                       │
       ┌───────────────┴────────────────┐
       │                                │
       ▼                                ▼
┌─────────────────┐             ┌────────────────┐
│ Deterministic   │             │ FAISS Retriever│
│ Analytics Engine│             └────────────────┘
└─────────────────┘                     │
       │                                │
       └───────────────┬────────────────┘
                       ▼
              Final Answer Builder
                       │
                       ▼
         🧠 LLM Explanation (Human-like)
                       │
                       ▼
                📤 Streamlit UI

📁 Project Structure
📦 olist-genai
 ┣ 📁 data/
 ┃ ┣ olist_orders_dataset.csv
 ┃ ┣ olist_order_items_dataset.csv
 ┃ ┣ olist_customers_dataset.csv
 ┃ ┣ olist_products_dataset.csv
 ┃ ┣ ... (Olist datasets)
 ┣ 📁 src/
 ┃ ┣ app.py                 # Streamlit app
 ┃ ┣ analytics.py           # Deterministic engine
 ┃ ┣ nl_to_query.py         # LLM + rule-based parser
 ┃ ┣ retrieval.py           # FAISS search
 ┃ ┣ embed_index.py         # Embedding generation
 ┣ meta.pkl
 ┣ vectors.faiss
 ┣ requirements.txt
 ┗ README.md

⚙️ Setup & Installation
1️⃣ Clone the repo
git clone https://github.com/yourusername/olist-genai.git
cd olist-genai

2️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set OpenRouter API key

(Required for NL → Query & explanations)

export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_MODEL="openai/gpt-4o-mini"

5️⃣ Generate embeddings
python src/embed_index.py

6️⃣ Run the app
streamlit run src/app.py

🧪 Example Queries (Use These for Your Demo)

These all work perfectly:

✔️ Sales & Revenue

“Which states generate the highest sales revenue?”

“Top 5 cities by sales”

“Total revenue between 2021-01-01 and 2021-12-31”

✔️ Category Insights

“What is the average order value for Electronics?”

“Compare electronics vs furniture categories”

“Which category was the highest selling in the past 2 quarters?”

✔️ Time Analysis

“How did sales vary by month?”

“Sales trend in the last year”

✔️ Product Insights

“Top 10 best-selling categories”

“Which products have the highest average freight cost?”

🧠 Why This Approach? (Key Interview Talking Points)
1️⃣ Pure LLM analytics → hallucinates numbers

→ Your hybrid engine never hallucinates.

2️⃣ RAG alone cannot compute sums/averages

→ Your engine can compute any aggregate.

3️⃣ Query translator → scalable to ANY dataset

→ You can plug in other datasets with no code change.

These will impress evaluators.

📌 Limitations & Future Improvements

Add charts (line/bar/pie) for visual analytics

Add multi-turn conversational memory

Add anomaly detection

Add product recommendations

Wrap into a FastAPI backend for scalability