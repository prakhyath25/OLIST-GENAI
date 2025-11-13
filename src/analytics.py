# src/analytics.py
import os
import pandas as pd
from datetime import datetime
import calendar
from typing import Dict, Any, Tuple, Optional, List

# env-path to the merged CSV (set DATA_CSV to your merged csv path if different)
DATA_CSV = os.getenv("DATA_CSV", "data/olist_combined_dataset.csv")

# simple in-process cache
_df_cache: Optional[pd.DataFrame] = None


def _load_df() -> pd.DataFrame:
    """
    Load the merged dataset lazily and normalize a few column names.
    Accepts either 'timestamp' or 'order_purchase_timestamp' as the datetime column.
    Normalizes category into column 'category' and ensures 'price' numeric.
    Raises FileNotFoundError if DATA_CSV does not exist.
    """
    global _df_cache
    if _df_cache is None:
        if not os.path.exists(DATA_CSV):
            raise FileNotFoundError(f"DATA_CSV not found at {DATA_CSV}")

        df = pd.read_csv(DATA_CSV, low_memory=False)

        # normalize timestamp column
        if "order_purchase_timestamp" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"order_purchase_timestamp": "timestamp"})
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)

        # ensure numeric price column exists
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        else:
            # create a price column if missing (all zeros) to avoid KeyErrors in analytics
            df["price"] = 0.0

        # normalize category column
        if "product_category_name_english" in df.columns:
            df["category"] = df["product_category_name_english"].fillna("Unknown")
        elif "product_category_name" in df.columns:
            df["category"] = df["product_category_name"].fillna("Unknown")
        elif "category" not in df.columns:
            df["category"] = "Unknown"

        # ensure customer_state/customer_city exist (create if missing)
        if "customer_state" not in df.columns:
            df["customer_state"] = df.get("customer_state", pd.Series([""] * len(df)))
        if "customer_city" not in df.columns:
            df["customer_city"] = df.get("customer_city", pd.Series([""] * len(df)))

        _df_cache = df

    return _df_cache


# ------------------------
# Time-range / quarter helpers
# ------------------------
def get_last_n_quarters_range(
    df: pd.DataFrame,
    n: int = 2,
    include_current_partial: bool = False,
    date_col: str = "timestamp",
) -> Tuple[datetime, datetime]:
    """
    Return (start_datetime, end_datetime) that cover the last n quarters relative to the latest timestamp in df.
    By default returns last n *completed* quarters (include_current_partial=False).

    Example: if latest timestamp is 2025-11-13 and include_current_partial=False and n=2,
    this returns the start of Q2 2025 and end of Q3 2025 (two completed quarters prior to current partial).
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in dataframe")
    last = df[date_col].max()
    if pd.isna(last):
        raise ValueError("No valid timestamps in dataframe")

    year = int(last.year)
    q = (last.month - 1) // 3 + 1  # quarter 1..4

    # If not including current partial quarter, move back to previous quarter
    if not include_current_partial:
        q -= 1
        if q == 0:
            q = 4
            year -= 1

    # compute start quarter/year
    start_q = q - (n - 1)
    start_year = year
    while start_q <= 0:
        start_q += 4
        start_year -= 1

    start_month = 3 * (start_q - 1) + 1
    start_date = datetime(start_year, start_month, 1, 0, 0, 0)

    # end = last second of (q,year)
    end_month = 3 * q
    last_day = calendar.monthrange(year, end_month)[1]
    end_date = datetime(year, end_month, last_day, 23, 59, 59)

    return start_date, end_date


def top_categories_by_revenue(
    df: pd.DataFrame,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    date_col: str = "timestamp",
    category_col: str = "category",
    value_col: str = "price",
    top_n: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregate total revenue by category for rows in [date_from, date_to].
    date_from/date_to may be None (meaning full dataset).
    Returns (agg_df_sorted_top_n, diagnostics_dict).

    Diagnostics includes filtered_rows, total_rows, min_date, max_date.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found")

    # ensure datetime dtype for date_col
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    mask = pd.Series(True, index=df.index)
    if date_from is not None:
        date_from = pd.to_datetime(date_from, errors="coerce")
        if pd.isna(date_from):
            raise ValueError("date_from could not be parsed")
        mask &= df[date_col] >= date_from
    if date_to is not None:
        date_to = pd.to_datetime(date_to, errors="coerce")
        if pd.isna(date_to):
            raise ValueError("date_to could not be parsed")
        mask &= df[date_col] <= date_to

    filtered = df.loc[mask].copy()
    diagnostics = {
        "filtered_rows": int(len(filtered)),
        "total_rows": int(len(df)),
        "min_date": None if df[date_col].isna().all() else df[date_col].min(),
        "max_date": None if df[date_col].isna().all() else df[date_col].max(),
    }

    if filtered.empty:
        return pd.DataFrame(), diagnostics

    filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce").fillna(0.0)

    agg = (
        filtered.groupby(category_col, dropna=False)[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "total_revenue"})
        .sort_values("total_revenue", ascending=False)
    )

    return agg.head(top_n), diagnostics


# ------------------------
# additional analytics helpers
# ------------------------
def avg_price_by_category(
    df: pd.DataFrame,
    category: str,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    date_col: str = "timestamp",
    category_col: str = "category",
    value_col: str = "price",
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Compute mean(price) and count for the given category in the optional date range.
    Returns (mean_price_or_None, diagnostics).
    """
    # ensure df loaded properly
    if category_col not in df.columns:
        raise KeyError(f"Category column '{category_col}' not found")

    # filter by category
    subset = df[df[category_col].astype(str).str.lower() == str(category).lower()].copy()

    # apply date filters if provided
    if date_from is not None:
        subset = subset[subset[date_col] >= pd.to_datetime(date_from, errors="coerce")]
    if date_to is not None:
        subset = subset[subset[date_col] <= pd.to_datetime(date_to, errors="coerce")]

    diagnostics = {"rows": int(len(subset))}

    if subset.empty:
        return None, diagnostics

    mean_price = float(pd.to_numeric(subset[value_col], errors="coerce").fillna(0.0).mean())
    diagnostics["mean_price"] = mean_price
    diagnostics["total_revenue"] = float(subset[value_col].sum())

    return mean_price, diagnostics


def compare_categories(
    df: pd.DataFrame,
    categories: List[str],
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    date_col: str = "timestamp",
    category_col: str = "category",
    value_col: str = "price",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compare mean price and total revenue between categories.
    categories should be a list of 2 or more category name strings.
    Returns (summary_df, diagnostics). summary_df has columns:
      ['category', 'mean_price', 'total_revenue', 'count']
    Diagnostics includes filtered_rows etc.
    """
    if not categories or len(categories) < 2:
        raise ValueError("Please provide at least two categories to compare")

    # ensure datetime dtype
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    mask = pd.Series(False, index=df.index)
    if date_from is not None:
        date_from = pd.to_datetime(date_from, errors="coerce")
        mask |= (df[date_col] >= date_from)
    if date_to is not None:
        date_to = pd.to_datetime(date_to, errors="coerce")
        mask |= (df[date_col] <= date_to)
    # If no date filters given, use all rows
    if date_from is None and date_to is None:
        filtered = df.copy()
    else:
        # apply both if provided (we want rows inside the interval)
        filtered = df.copy()
        if date_from is not None:
            filtered = filtered[filtered[date_col] >= date_from]
        if date_to is not None:
            filtered = filtered[filtered[date_col] <= date_to]

    diagnostics = {
        "filtered_rows": int(len(filtered)),
        "total_rows": int(len(df)),
    }

    rows = []
    for cat in categories:
        subset = filtered[filtered[category_col].astype(str).str.lower() == str(cat).lower()].copy()
        count = int(len(subset))
        total_revenue = float(subset[value_col].sum()) if count > 0 else 0.0
        mean_price = float(subset[value_col].mean()) if count > 0 else float("nan")
        rows.append({"category": cat, "mean_price": mean_price, "total_revenue": total_revenue, "count": count})

    summary_df = pd.DataFrame(rows).sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return summary_df, diagnostics


def monthly_sales_by_month(
    df: pd.DataFrame,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    date_col: str = "timestamp",
    value_col: str = "price",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Return a DataFrame grouping by year-month and summing value_col.
    Columns: ['year', 'month', 'year_month', 'total_revenue'] sorted ascending by year_month.
    """
    d = df.copy()
    if date_col in d.columns and not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    if date_from is not None:
        d = d[d[date_col] >= pd.to_datetime(date_from, errors="coerce")]
    if date_to is not None:
        d = d[d[date_col] <= pd.to_datetime(date_to, errors="coerce")]

    diagnostics = {"filtered_rows": int(len(d)), "total_rows": int(len(df))}

    if d.empty:
        return pd.DataFrame(), diagnostics

    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    d["year_month"] = d[date_col].dt.strftime("%Y-%m")
    agg = d.groupby(["year", "month", "year_month"], sort=True)[value_col].sum().reset_index().rename(columns={value_col: "total_revenue"})
    agg = agg.sort_values(["year", "month"]).reset_index(drop=True)
    return agg, diagnostics


# ------------------------
# Existing run_aggregate used by the app (enhanced)
# ------------------------
def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a small set of supported filters: date_from, date_to, category, customer_state
    """
    if not filters:
        return df
    d = df
    if filters.get("date_from"):
        d = d[d["timestamp"] >= pd.to_datetime(filters["date_from"], errors="coerce")]
    if filters.get("date_to"):
        d = d[d["timestamp"] <= pd.to_datetime(filters["date_to"], errors="coerce")]
    if filters.get("category"):
        d = d[d["category"].astype(str).str.lower() == str(filters["category"]).lower()]
    if filters.get("customer_state"):
        d = d[d["customer_state"].astype(str).str.upper() == str(filters["customer_state"]).upper()]
    return d


def run_aggregate(query: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Run simple aggregate queries. Supports date filters in query['filter'].
    Returns (ok, result_dict) where result_dict has 'value' or 'table' (pandas.DataFrame) or 'error'.
    Query schema (expected):
      {
        "action":"aggregate",
        "agg":"sum"|"mean"|"count",
        "column":"price"|"freight"|"order_id",
        "group_by":"customer_state"|"customer_city"|"category" or None,
        "filter": {"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", ...}
      }
    """
    try:
        df = _load_df()
    except Exception as e:
        return False, {"error": f"Failed to load data: {e}"}

    filters = query.get("filter", {}) or {}
    df_filtered = apply_filters(df, filters)

    agg = (query.get("agg") or "").lower()
    col = query.get("column") or "price"
    group_by = query.get("group_by", None)

    # safe checks
    if col not in df_filtered.columns:
        return False, {"error": f"Column '{col}' not found in data."}
    if group_by is not None and group_by not in df_filtered.columns:
        return False, {"error": f"group_by column '{group_by}' not found in data."}

    try:
        if agg in ("mean", "avg"):
            if group_by:
                res = df_filtered.groupby(group_by)[col].mean().reset_index().sort_values(by=col, ascending=False)
                return True, {"table": res, "summary": f"mean_{col}_by_{group_by}"}
            else:
                val = float(df_filtered[col].mean())
                return True, {"value": val}
        elif agg == "sum":
            if group_by:
                res = df_filtered.groupby(group_by)[col].sum().reset_index().sort_values(by=col, ascending=False)
                return True, {"table": res, "summary": f"sum_{col}_by_{group_by}"}
            else:
                val = float(df_filtered[col].sum())
                return True, {"value": val}
        elif agg == "count":
            if group_by:
                res = df_filtered.groupby(group_by).size().reset_index(name="count").sort_values(by="count", ascending=False)
                return True, {"table": res, "summary": f"count_by_{group_by}"}
            else:
                return True, {"value": int(len(df_filtered))}
        else:
            return False, {"error": "Unsupported aggregation: " + str(agg)}
    except Exception as e:
        return False, {"error": f"Aggregation failed: {e}"}
