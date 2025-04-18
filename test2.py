import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

product_features = np.load('models/product_features.npy')
product_scaler = joblib.load('models/product_scaler.pkl')
encoded_df = joblib.load('models/encoded_df.pkl')
products_df = joblib.load('models/products_df.pkl')

products_df = products_df.drop_duplicates(subset='product_id').reset_index(drop=True)

encoded_df = pd.get_dummies(products_df['product_category_name_english'])
encoded_df['price'] = products_df['price'].values
encoded_df = encoded_df.reset_index(drop=True)

assert encoded_df.shape[0] == product_features.shape[0], "Mismatch between encoded features and product features."

st.sidebar.title("🔍 Product Filters")
category_options = products_df['product_category_name_english'].dropna().unique()
selected_category = st.sidebar.selectbox("Select Category", sorted(category_options))

price_min = float(products_df['price'].min())
price_max = float(products_df['price'].max())
selected_price = st.sidebar.slider("Select Max Price", min_value=price_min, max_value=price_max, value=price_max)

st.title("🛍️ Content-Based Product Recommender")
st.markdown("Get product recommendations based on category and price range.")

# Filter based on user selection
if selected_category not in encoded_df.columns:
    st.warning("Selected category not found in encoded data.")
else:
    # Create user input vector
    user_input_vector = np.zeros(encoded_df.shape[1])
    user_input_vector[encoded_df.columns.get_loc(selected_category)] = 1
    user_input_vector[-1] = selected_price

    # Scale and compute similarity
    scaled_input = product_scaler.transform([user_input_vector])
    similarities = cosine_similarity(scaled_input, product_features).flatten()

    # Get top similar indices and filter those that match user constraints
    sorted_indices = np.argsort(similarities)[::-1]

    recommended_products = []
    for idx in sorted_indices:
        product = products_df.iloc[idx]
        if (
            product['product_category_name_english'] == selected_category and
            product['price'] <= selected_price
        ):
            recommended_products.append(product)
        if len(recommended_products) == 5:
            break

    if recommended_products:
        st.subheader("🧠 Recommended Products:")
        st.table(pd.DataFrame(recommended_products)[['product_id', 'product_category_name_english', 'price']])
    else:
        st.warning("No similar products found in selected filters.")
