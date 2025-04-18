import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-Commerce ML App", page_icon="🛒")

st.sidebar.title("📋 Navigation")
app_mode = st.sidebar.radio("Choose an option", ["🏠 Home", "👥 Customer Segmentation", "🛍️ Product Recommendation"])

if app_mode == "🏠 Home":
    st.title("🛒 E-Commerce Intelligence App")
    st.markdown("""
    Welcome to the E-Commerce Intelligence App powered by Machine Learning!  
    Use the sidebar to:
    - Predict customer behavior using **KMeans clustering**
    - Get personalized **product recommendations** based on content similarity  
    """)
    st.image(
        "https://bloomidea.com/sites/default/files/styles/og_image/public/blog/Tipos%20de%20come%CC%81rcio%20electro%CC%81nico_0.png?itok=jC9MlQZq",
        use_column_width=True
    )

elif app_mode == "👥 Customer Segmentation":
    st.title("🧠 Customer Segmentation Predictor")
    st.markdown("Enter customer behavior data to predict the customer segment:")

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)

    payment = st.number_input("Average Payment Value (R$)", min_value=0.0, step=1.0)
    orders = st.number_input("Number of Orders", min_value=0)
    review = st.slider("Average Review Score", 0.0, 5.0, 4.0)
    freight = st.number_input("Average Freight Value (R$)", min_value=0.0, step=1.0)

    if st.button("Predict Segment"):
        user_input = np.array([[payment, orders, review, freight]])
        user_scaled = scaler.transform(user_input)
        cluster = model.predict(user_scaled)[0]

        st.success(f"This customer belongs to **Cluster {cluster}**")
        if cluster == 0:
            st.info("🟢 Likely a loyal customer with high satisfaction.")
        elif cluster == 1:
            st.info("🟡 New or budget-conscious customer.")
        elif cluster == 2:
            st.info("🔴 High freight cost, low reviews — risky customer.")
        else:
            st.info("🔵 Cluster behavior under analysis.")

elif app_mode == "🛍️ Product Recommendation":
    st.title("🛍️ Content-Based Product Recommender")
    st.markdown("Get product recommendations based on category and price range.")

    product_features = np.load('models/product_features.npy')
    product_scaler = joblib.load('models/product_scaler.pkl')
    encoded_df = joblib.load('models/encoded_df.pkl')
    products_df = pd.read_pickle('models/products_df.pkl')  # ✅ FIXED HERE

    products_df = products_df.drop_duplicates(subset='product_id').reset_index(drop=True)

    encoded_df = pd.get_dummies(products_df['product_category_name_english'])
    encoded_df['price'] = products_df['price'].values
    encoded_df = encoded_df.reset_index(drop=True)

    assert encoded_df.shape[0] == product_features.shape[0], "Mismatch between encoded features and product features."

    category_options = products_df['product_category_name_english'].dropna().unique()
    selected_category = st.sidebar.selectbox("Select Category", sorted(category_options))

    price_min = float(products_df['price'].min())
    price_max = float(products_df['price'].max())
    selected_price = st.sidebar.slider("Select Max Price", min_value=price_min, max_value=price_max, value=price_max)

    if selected_category not in encoded_df.columns:
        st.warning("Selected category not found in encoded data.")
    else:
        user_input_vector = np.zeros(encoded_df.shape[1])
        user_input_vector[encoded_df.columns.get_loc(selected_category)] = 1
        user_input_vector[-1] = selected_price

        scaled_input = product_scaler.transform([user_input_vector])
        similarities = cosine_similarity(scaled_input, product_features).flatten()
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
