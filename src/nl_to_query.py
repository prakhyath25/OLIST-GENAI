# src/nl_to_query.py
import os
import json
import re
from typing import Dict, Any, Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

def _call_openrouter(nl_text: str, system_prompt: str = None) -> Optional[Dict[str, Any]]:
    """Call OpenRouter chat completion to convert NL -> structured JSON query.
       Returns parsed JSON dict on success or None on failure.
    """
    if not OPENROUTER_API_KEY:
        return None
    import requests
    url = "https://openrouter.ai/api/v1/chat/completions"  # prefer api.openrouter.ai if DNS works
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if system_prompt is None:
        system_prompt = (
            "You are a translator. Convert the user's question into a JSON object describing a SAFE structured query "
            "using only these columns: price, freight, customer_state, customer_city, category, timestamp. "
            "Return EXACT JSON (no explanation). Schema examples: "
            '{"action":"aggregate","agg":"sum","column":"price","group_by":"customer_state","filter":{"date_from":"YYYY-MM-DD","date_to":"YYYY-MM-DD","category":"...","customer_state":"..."} } '
            'or {"action":"describe"} if no numeric aggregate requested.'
        )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_text}
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if "application/json" not in resp.headers.get("content-type", ""):
            return None
        data = resp.json()
        # Try to extract content robustly
        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text")
        # parse as JSON
        return json.loads(content.strip())
    except Exception:
        return None

def _rule_based_parse(nl_text: str) -> Dict[str, Any]:
    """Simple rule-based fallback to detect aggregate intent and group_by fields."""
    t = nl_text.lower()
    # detect aggregate
    agg = None
    if re.search(r"\b(total|sum|revenue|sales)\b", t):
        agg = "sum"
        column = "price"
    elif re.search(r"\b(average|mean|avg)\b", t):
        agg = "mean"
        column = "price"
    elif re.search(r"\b(count|how many|number of)\b", t):
        agg = "count"
        column = "order_id"  # count of rows
    else:
        return {"action": "describe"}

    # group_by detection
        # group_by detection
        
    group_by = None
    if "state" in t or "states" in t or "customer_state" in t or "state-wise" in t:
        group_by = "customer_state"
    elif "city" in t or "cities" in t or "customer_city" in t:
        group_by = "customer_city"
    elif "category" in t:
        group_by = "category"
    elif re.search(r"\bby\s+month\b|\bmonthly\b|\bper month\b", t):
        group_by = "month"
    elif re.search(r"\bby\s+year\b|\byearly\b|\bper year\b", t):
        group_by = "year"
    elif re.search(r"\bby\s+quarter|\bquarterly\b", t):
        group_by = "quarter"


    # date detection (simple)
    date_from = None
    date_to = None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        date_from = m.group(1)
    # try to find "between A and B"
    m2 = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", t)
    if m2:
        date_from, date_to = m2.group(1), m2.group(2)

    return {
        "action": "aggregate",
        "agg": agg,
        "column": column,
        "group_by": group_by,
        "filter": {
            "date_from": date_from,
            "date_to": date_to
        }
    }

def nl_to_query(nl_text: str) -> Dict[str, Any]:
    """
    Convert natural language to a structured query dict.
    Tries OpenRouter if configured; falls back to a rule-based parser.
    """
    # try LLM first
    llm_result = _call_openrouter(nl_text)
    if llm_result and isinstance(llm_result, dict):
        # Basic validation: required keys for aggregate
        if llm_result.get("action") == "aggregate":
            # allowlist/whitelist checks: only accept known column names and aggs
            allowed_aggs = {"sum", "mean", "avg", "count"}
            allowed_columns = {"price", "freight", "order_id"}
            if llm_result.get("agg") and llm_result.get("agg").lower() in allowed_aggs:
                # sanitize column
                col = llm_result.get("column", "").lower()
                if col in {"price", "freight", "order_id"}:
                    return llm_result
        elif llm_result.get("action") == "describe":
            return {"action": "describe"}
    # fallback
    return _rule_based_parse(nl_text)
