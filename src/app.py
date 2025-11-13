# src/app.py
import os
import json
import re
import requests
import streamlit as st
from retrieval import Retriever
# add near other imports
import pandas as pd


# new helpers
from nl_to_query import nl_to_query
from analytics import (
    run_aggregate,
    _load_df,
    get_last_n_quarters_range,
    top_categories_by_revenue,
    compare_categories,            # <-- import the compare helper
)

# --- CONFIG (env override) ---
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful data analyst. Use only the supplied context rows when answering; do not hallucinate."
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# OpenRouter endpoint override
OPENROUTER_ENDPOINT = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")

st.set_page_config(page_title="Olist GenAI — Hybrid", layout="wide")
st.title("Olist GenAI — Hybrid: Deterministic analytics + LLM for explanation")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": SYSTEM_PROMPT}]

# instantiate retriever
try:
    retriever = Retriever()
except Exception as e:
    st.error(f"Failed to initialize Retriever: {e}")
    st.stop()


def call_openrouter_explain(system_prompt: str, user_prompt: str, model: str, max_tokens: int = 200, temperature: float = 0.0):
    """
    Calls OpenRouter to produce a friendly explanation. Returns (ok, content, raw_resp_or_error).
    """
    if not OPENROUTER_API_KEY:
        return False, "OPENROUTER_API_KEY not set.", None

    url = OPENROUTER_ENDPOINT
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
    except Exception as e:
        return False, f"Network/connection error calling OpenRouter: {e}", None

    content_type = resp.headers.get("content-type", "")
    text = resp.text or ""
    if "application/json" not in content_type:
        snippet = text[:2000].replace("\n", " ")
        return False, f"OpenRouter returned non-JSON response (content-type={content_type}). Snippet:\n{snippet}", resp
    try:
        data = resp.json()
    except Exception as e:
        return False, f"OpenRouter returned invalid JSON: {e}. Raw text: {text[:2000]}", resp

    if isinstance(data, dict) and data.get("error"):
        return False, f"OpenRouter error: {data.get('error')}", resp

    try:
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice.get("message"), dict):
                content = choice["message"].get("content") or ""
            else:
                content = choice.get("text") or str(choice)
            return True, content.strip(), resp
        else:
            return False, "No choices in response", resp
    except Exception as e:
        return False, f"Failed to parse response: {e}", resp


# UI input
with st.form("query_form", clear_on_submit=True):
    user_q = st.text_input("Ask a question about the dataset", "")
    k = st.number_input("retrieval k (support rows)", min_value=1, max_value=1000, value=6, step=1)
    submitted = st.form_submit_button("Send")

if submitted and user_q.strip():
    st.session_state['messages'].append({"role": "user", "content": user_q})

    # 1) retrieval for provenance
    hits = retriever.retrieve(user_q, k=k)
    context_text = "\n\n".join([f"Row index:{h.get('index','')} | {h.get('text','')}" for h in hits])

    # 2) NL -> structured query (safe)
    structured = nl_to_query(user_q)
    st.write("**Structured query (interpreted):**")
    st.json(structured)

    answer_text = None
    debug_msgs = []

    # Detect "past N quarters" directly in NL (override structured path if matched)
    q_match = re.search(r'(past|last|previous)\s+(\d+)\s+quarters?', user_q, re.I)
    if q_match:
        n_quarters = int(q_match.group(2))
        try:
            df = _load_df()
            start_dt, end_dt = get_last_n_quarters_range(df, n=n_quarters, include_current_partial=False)
            agg_df, diag = top_categories_by_revenue(df, date_from=start_dt, date_to=end_dt, top_n=10)
            if agg_df.empty:
                answer_text = f"No sales rows in the last {n_quarters} completed quarters. Diagnostics: {diag}"
            else:
                top = agg_df.iloc[0]
                # build deterministic summary
                deterministic_summary = (
                    f"Top category in the last {n_quarters} completed quarters: {top['category']} "
                    f"with total revenue = ${top['total_revenue']:.2f}."
                )
                det_table = agg_df.head(5).to_dict(orient='records')
                answer_text = deterministic_summary + "\n\nTop categories (top 5):\n" + json.dumps(det_table, indent=2)

                # append provenance
                supporting_indices = [h.get("index") for h in hits]
                answer_text += "\n\nSupporting row indices (retrieved): " + ", ".join([str(i) for i in supporting_indices])

                # optionally LLM explanation
                if OPENROUTER_API_KEY:
                    expl_prompt = (
                        f"User question: {user_q}\n\n"
                        f"Deterministic result summary: {answer_text}\n\n"
                        "Write a concise 2-4 sentence explanation, mention that numbers were computed deterministically and list top 3 supporting row indices."
                    )
                    ok_llm, llm_resp, _ = call_openrouter_explain(SYSTEM_PROMPT, expl_prompt, OPENROUTER_MODEL, max_tokens=150)
                    if ok_llm:
                        answer_text = llm_resp
                    else:
                        debug_msgs.append("LLM explanation failed: " + str(llm_resp))
        except Exception as e:
            debug_msgs.append(f"Quarter-analytics failed: {e}")

    # If not handled by quarter-detection, handle via structured query / aggregate / compare
    if answer_text is None:
        # #######################
        # NEW: handle compare action (mean difference etc.)
        # #######################
        if structured.get("action") == "compare" or re.search(r"\b(compare|difference|versus|vs\.?)\b", user_q, re.I):
            # Expect structured to include categories list, or parse them heuristically from NL
            categories = structured.get("categories")
            if not categories:
                # naive extraction of the two category tokens from user question
                # e.g. "electronics and furniture" -> ['electronics', 'furniture']
                cat_matches = re.findall(r"\b([A-Za-z_ ]+?)\b\s+(?:and|vs|versus|vs\.)\s+\b([A-Za-z_ ]+?)\b", user_q, re.I)
                if cat_matches:
                    # take first tuple
                    a, b = cat_matches[0]
                    categories = [a.strip(), b.strip()]
            # fallback: try to find two words after 'between' or 'difference between'
            if not categories:
                m = re.search(r"difference between\s+([A-Za-z_ ]+?)\s+and\s+([A-Za-z_ ]+?)", user_q, re.I)
                if m:
                    categories = [m.group(1).strip(), m.group(2).strip()]

            if not categories or len(categories) < 2:
                answer_text = "I couldn't identify two categories to compare. Please ask 'Compare X and Y' or 'difference between X and Y'."
            else:
                try:
                    df = _load_df()
                    summary_df, diag = compare_categories(df, categories[:2])  # use first two
                    if summary_df.empty:
                        answer_text = f"No rows found for the requested categories: {categories[:2]}. Diagnostics: {diag}"
                    else:
                        # format a readable answer
                        row0 = summary_df.iloc[0].to_dict()
                        row1 = summary_df.iloc[1].to_dict() if len(summary_df) > 1 else None
                        # determine which is electronics/furniture-like: rely on provided labels
                        cat_a = summary_df.iloc[0]["category"]
                        cat_b = summary_df.iloc[1]["category"] if row1 else None

                        mean_a = float(summary_df.loc[summary_df['category'].str.lower() == str(categories[0]).lower(), 'mean_price'].values[0]) \
                            if (summary_df['category'].str.lower() == str(categories[0]).lower()).any() else float("nan")
                        mean_b = float(summary_df.loc[summary_df['category'].str.lower() == str(categories[1]).lower(), 'mean_price'].values[0]) \
                            if (summary_df['category'].str.lower() == str(categories[1]).lower()).any() else float("nan")

                        # fallback: use the summary_df ordered by total_revenue if mean lookups failed
                        if pd.isna(mean_a) or pd.isna(mean_b):
                            mean_a = float(summary_df.iloc[0]['mean_price'])
                            mean_b = float(summary_df.iloc[1]['mean_price']) if len(summary_df) > 1 else float("nan")

                        mean_diff = None
                        if not pd.isna(mean_a) and not pd.isna(mean_b):
                            mean_diff = mean_a - mean_b

                        # human readable
                        readable = (
                            f"{categories[0].title()} average price ≈ {mean_a:.2f}, "
                            f"{categories[1].title()} average price ≈ {mean_b:.2f}."
                        )
                        if mean_diff is not None:
                            readable += f" Difference ( {categories[0].title()} - {categories[1].title()} ) ≈ {mean_diff:.2f}."

                        # also include total revenues & counts
                        total_rev_a = float(summary_df.loc[summary_df['category'].str.lower() == str(categories[0]).lower(), 'total_revenue'].values[0]) \
                            if (summary_df['category'].str.lower() == str(categories[0]).lower()).any() else float(summary_df.iloc[0]['total_revenue'])
                        total_rev_b = float(summary_df.loc[summary_df['category'].str.lower() == str(categories[1]).lower(), 'total_revenue'].values[0]) \
                            if (summary_df['category'].str.lower() == str(categories[1]).lower()).any() else float(summary_df.iloc[1]['total_revenue']) if len(summary_df) > 1 else 0.0

                        readable += f" Total revenue: {categories[0].title()} = ${total_rev_a:.2f}, {categories[1].title()} = ${total_rev_b:.2f}."

                        answer_text = readable
                        # append supporting provenance
                        supporting_indices = [h.get("index") for h in hits]
                        answer_text += "\n\nSupporting row indices (retrieved): " + ", ".join([str(i) for i in supporting_indices])

                        # optionally ask LLM to write a polished sentence
                        if OPENROUTER_API_KEY:
                            expl_prompt = (
                                f"User question: {user_q}\n\n"
                                f"Deterministic summary: {answer_text}\n\n"
                                "Write a concise 1-2 sentence user-facing explanation (no tables). Mention that numbers were computed deterministically."
                            )
                            ok_llm, llm_resp, _ = call_openrouter_explain(SYSTEM_PROMPT, expl_prompt, OPENROUTER_MODEL, max_tokens=120)
                            if ok_llm:
                                answer_text = llm_resp
                            else:
                                debug_msgs.append("LLM explanation failed: " + str(llm_resp))
                except Exception as e:
                    debug_msgs.append(f"Compare action failed: {e}")
        # #######################
        # Existing aggregate path
        # #######################
        elif structured.get("action") == "aggregate":
            ok, result = run_aggregate(structured)
            if not ok:
                debug_msgs.append("Analytics error: " + str(result.get("error")))
            else:
                if "value" in result:
                    answer_text = f"Result: **{result['value']}** for {structured.get('agg')}({structured.get('column')})."
                elif "table" in result:
                    df_res = result["table"]
                    st.markdown("### Numeric result (top rows)")
                    st.dataframe(df_res.head(20))
                    if structured.get("group_by"):
                        try:
                            top_row = df_res.iloc[0]
                            gb = structured.get("group_by")
                            metric = structured.get("column")
                            answer_text = f"Top {gb}: **{top_row[gb]}** with {structured.get('agg')}({metric}) = **{top_row[metric]}**."
                        except Exception:
                            answer_text = "Aggregate returned a table but failed to extract top row."
                    else:
                        answer_text = f"Result: **{list(df_res.iloc[0].values)}**."
                else:
                    answer_text = "Aggregate ran but no result returned."

                supporting_indices = [h.get("index") for h in hits]
                answer_text += "\n\nSupporting row indices (retrieved): " + ", ".join([str(i) for i in supporting_indices])

                if OPENROUTER_API_KEY:
                    expl_prompt = (
                        f"User question: {user_q}\n\n"
                        f"Structured query: {json.dumps(structured)}\n\n"
                        f"Deterministic result summary: {answer_text}\n\n"
                        "Write a concise 2-4 sentence explanation and mention top 3 supporting row indices."
                    )
                    ok_llm, llm_resp, _ = call_openrouter_explain(SYSTEM_PROMPT, expl_prompt, OPENROUTER_MODEL, max_tokens=150)
                    if ok_llm:
                        answer_text = llm_resp
                    else:
                        debug_msgs.append("LLM explanation failed: " + str(llm_resp))

        else:
            # descriptive / fallback path uses LLM (if available) to summarize retrieved context rows
            if OPENROUTER_API_KEY:
                expl_prompt = (
                    f"User question: {user_q}\n\n"
                    f"Context rows:\n{context_text}\n\n"
                    "Summarize in 2-4 sentences and list up to 3 supporting row indices. If insufficient, say so."
                )
                ok_llm, llm_resp, _ = call_openrouter_explain(SYSTEM_PROMPT, expl_prompt, OPENROUTER_MODEL, max_tokens=200)
                if ok_llm:
                    answer_text = llm_resp
                else:
                    debug_msgs.append("LLM summarize failed: " + str(llm_resp))
                    answer_text = "I don't have enough data in the retrieved rows to answer exactly."
            else:
                answer_text = "Descriptive mode: showing retrieved context rows. For precise numeric answers, ask a question with 'total', 'sum', 'average', or 'by state'."

    if not answer_text:
        answer_text = "No answer produced. Debug: " + "; ".join(debug_msgs)

    st.session_state['messages'].append({"role": "assistant", "content": answer_text})

    st.markdown("### Answer")
    st.write(answer_text)

    st.markdown("### Supporting rows (retrieved)")
    for i, h in enumerate(hits):
        st.markdown(f"- (score {h.get('_score', 0):.2f}) index:{h.get('index','')} — {h.get('text','')[:350]}")

# conversation sidebar unchanged
st.sidebar.header("Conversation")
for m in st.session_state['messages']:
    role = m['role']
    content = m['content']
    if role == 'system':
        continue
    st.sidebar.markdown(f"**{role}:** {content[:200]}")
