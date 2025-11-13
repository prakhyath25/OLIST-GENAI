# src/data_prep.py
import os
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), "data")

def load_all():
    # adjust filenames if needed
    orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv", parse_dates=['order_purchase_timestamp'], low_memory=False)
    customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv", low_memory=False)
    order_items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv", low_memory=False)
    products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv", low_memory=False)
    category = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv", low_memory=False)

    # join products with categories translation if available
    products = products.merge(category, how='left', left_on='product_category_name', right_on='product_category_name', suffixes=('', '_trans'))

    # example light join: order_items + products + orders + customers
    df = order_items.merge(products[['product_id','product_name_lenght','product_description_lenght','product_category_name','product_category_name_english']],
                             on="product_id", how='left')
    df = df.merge(orders[['order_id','order_purchase_timestamp','customer_id']], on='order_id', how='left')
    df = df.merge(customers[['customer_id','customer_unique_id','customer_city','customer_state']], on='customer_id', how='left')

    # basic cleaning
    df['product_category_name_english'] = df['product_category_name_english'].fillna('Unknown')
    df['text'] = df.apply(lambda r: f"order_id:{r.order_id} product:{str(r.product_id)} category:{r.product_category_name_english} price:{r.price} freight:{r.freight_value} customer_city:{r.customer_city} customer_state:{r.customer_state}", axis=1)
    # ensure datetime + numeric price
    df['order_purchase_timestamp'] = pd.to_datetime(
    df['order_purchase_timestamp'], errors='coerce'
    )
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)

# If your category column is named differently, ensure canonical name:
    if 'product_category_name_english' not in df.columns and 'product_category_name' in df.columns:
        df['product_category_name_english'] = df['product_category_name'].fillna('Unknown')
    else:
        df['product_category_name_english'] = df['product_category_name_english'].fillna('Unknown')

    return df

if __name__ == "__main__":
    df = load_all()
    print("Rows:", len(df))
    print(df.head())
