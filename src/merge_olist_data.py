import pandas as pd
import os

# set base path
base = "data"

# load CSVs
customers = pd.read_csv(os.path.join(base, "olist_customers_dataset.csv"))
orders = pd.read_csv(os.path.join(base, "olist_orders_dataset.csv"))
items = pd.read_csv(os.path.join(base, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(base, "olist_products_dataset.csv"))
translation = pd.read_csv(os.path.join(base, "product_category_name_translation.csv"))

# merge step-by-step
df = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(items, on="order_id", how="left")
    .merge(products, on="product_id", how="left")
    .merge(translation, on="product_category_name", how="left")
)

# select useful columns
df = df[[
    "order_id",
    "order_purchase_timestamp",
    "price",
    "freight_value",
    "customer_city",
    "customer_state",
    "product_category_name_english"
]].rename(columns={
    "order_purchase_timestamp": "timestamp",
    "freight_value": "freight",
    "product_category_name_english": "category"
})

# convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# save combined dataset
output_path = os.path.join(base, "olist_combined_dataset.csv")
df.to_csv(output_path, index=False)

print(f"✅ Combined dataset saved to {output_path}")
print(df.head())
