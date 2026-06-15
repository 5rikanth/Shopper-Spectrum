"""
Shopper Spectrum — Customer Segmentation & Product Recommendations
Streamlit Application

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────
# Load Models (cached)
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_artifacts():
    """Load the trained KMeans model, scaler, and item-similarity matrix."""
    try:
        kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        item_sim_df = joblib.load(os.path.join(MODELS_DIR, "item_similarity.pkl"))
        rfm_data = pd.read_csv(os.path.join(MODELS_DIR, "rfm_data.csv"))
        return kmeans, scaler, item_sim_df, rfm_data
    except FileNotFoundError as e:
        st.error(
            f"Model file not found: {e}. "
            "Please run the analysis notebook first to generate the model files "
            "in the 'models/' directory."
        )
        st.stop()


kmeans_model, scaler, item_sim_df, rfm_data = load_artifacts()

# ──────────────────────────────────────────────────────────────────────────
# Derive Cluster -> Segment label mapping from the saved RFM data
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_label_map(rfm_data: pd.DataFrame) -> dict:
    """Recreate the Cluster ID -> Business Segment Name mapping."""
    if "Segment" in rfm_data.columns and "Cluster" in rfm_data.columns:
        mapping = (
            rfm_data.groupby("Cluster")["Segment"]
            .first()
            .to_dict()
        )
        return mapping
    # Fallback: derive from cluster centroid characteristics
    cluster_means = rfm_data.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    label_map = {}
    label_map[cluster_means["Monetary"].idxmax()] = "High-Value"
    remaining = [c for c in cluster_means.index if c not in label_map]
    label_map[cluster_means.loc[remaining, "Recency"].idxmax()] = "At-Risk"
    remaining = [c for c in cluster_means.index if c not in label_map]
    label_map[cluster_means.loc[remaining, "Frequency"].idxmax()] = "Regular"
    remaining = [c for c in cluster_means.index if c not in label_map]
    for c in remaining:
        label_map[c] = "Occasional"
    return label_map


label_map = get_label_map(rfm_data)

SEGMENT_INFO = {
    "High-Value": {
        "emoji": "💎",
        "color": "#E74C3C",
        "description": "Recent, frequent, and high-spending customers. Your VIPs!",
        "action": "Offer exclusive rewards, early access to new products, and dedicated support to retain loyalty."
    },
    "Regular": {
        "emoji": "🟢",
        "color": "#2ECC71",
        "description": "Consistent customers with healthy purchase frequency and spend.",
        "action": "Encourage upgrades with loyalty points and personalized cross-sell offers."
    },
    "Occasional": {
        "emoji": "🔵",
        "color": "#3498DB",
        "description": "Infrequent buyers with lower overall spend.",
        "action": "Re-engage with awareness campaigns, discounts, and product discovery emails."
    },
    "At-Risk": {
        "emoji": "🟠",
        "color": "#F39C12",
        "description": "Customers who haven't purchased recently and may be churning.",
        "action": "Launch win-back campaigns with personalized discounts and reminders."
    },
}

# ──────────────────────────────────────────────────────────────────────────
# Recommendation Function
# ──────────────────────────────────────────────────────────────────────────
def get_recommendations(product_name: str, top_n: int = 5):
    """Return top_n similar products for a given product name using
    item-based collaborative filtering (cosine similarity)."""
    product_name = product_name.strip().upper()

    if product_name in item_sim_df.index:
        matched_product = product_name
    else:
        # Fuzzy/partial match fallback
        matches = [p for p in item_sim_df.index if product_name in p]
        if not matches:
            return None, []
        matched_product = matches[0]

    scores = item_sim_df[matched_product].drop(matched_product).sort_values(ascending=False)
    top_items = scores.head(top_n)
    return matched_product, list(zip(top_items.index, top_items.values))


# ──────────────────────────────────────────────────────────────────────────
# Segmentation Prediction Function
# ──────────────────────────────────────────────────────────────────────────
def predict_segment(recency: float, frequency: float, monetary: float):
    """Predict the customer segment given RFM values."""
    input_df = pd.DataFrame([[recency, frequency, monetary]],
                             columns=["Recency", "Frequency", "Monetary"])
    input_scaled = scaler.transform(input_df)
    cluster = kmeans_model.predict(input_scaled)[0]
    segment = label_map.get(cluster, f"Cluster {cluster}")
    return segment, cluster


# ──────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown("**Customer Segmentation & Product Recommendations**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🎯 Product Recommendations", "👥 Customer Segmentation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses **RFM Analysis + KMeans Clustering** for customer "
    "segmentation and **Item-Based Collaborative Filtering** for product "
    "recommendations, built on real e-commerce transaction data."
)

# ──────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation and Product Recommendations in E-Commerce")

    st.markdown("""
    Welcome to **Shopper Spectrum** — an analytics-driven application that helps
    online retailers understand their customers and recommend the right products.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Product Recommendation Module")
        st.markdown(
            "Enter a product name and instantly get the **top 5 similar products** "
            "based on collaborative filtering — perfect for cross-sell and "
            "'customers also bought' style suggestions."
        )

    with col2:
        st.markdown("### 👥 Customer Segmentation Module")
        st.markdown(
            "Enter a customer's **Recency, Frequency, and Monetary (RFM)** values "
            "to instantly predict which customer segment they belong to — "
            "High-Value, Regular, Occasional, or At-Risk."
        )

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(rfm_data):,}")
    c2.metric("Products in Catalogue", f"{item_sim_df.shape[0]:,}")
    c3.metric("Avg. Monetary Value", f"£{rfm_data['Monetary'].mean():,.0f}")
    c4.metric("Avg. Purchase Frequency", f"{rfm_data['Frequency'].mean():.1f}")

    st.markdown("### Customer Segment Distribution")
    seg_counts = rfm_data["Segment"].value_counts() if "Segment" in rfm_data.columns else None
    if seg_counts is not None:
        seg_df = seg_counts.reset_index()
        seg_df.columns = ["Segment", "Count"]
        st.bar_chart(seg_df.set_index("Segment"))

# ──────────────────────────────────────────────────────────────────────────
# PRODUCT RECOMMENDATION PAGE
# ──────────────────────────────────────────────────────────────────────────
elif page == "🎯 Product Recommendations":
    st.title("🎯 Product Recommendation Engine")
    st.markdown(
        "Enter a product name below to find the **top 5 similar products** "
        "based on customer co-purchase patterns (item-based collaborative filtering)."
    )

    st.markdown("---")

    # Provide a sample list for convenience
    all_products = sorted(item_sim_df.index.tolist())

    input_method = st.radio(
        "How would you like to enter the product?",
        ["Select from list", "Type product name"],
        horizontal=True
    )

    product_input = None
    if input_method == "Select from list":
        product_input = st.selectbox(
            "Choose a product:",
            options=all_products,
            index=0 if all_products else None
        )
    else:
        product_input = st.text_input(
            "Type a product name (or part of it):",
            placeholder="e.g. JUMBO BAG RED RETROSPOT"
        )

    if st.button("🔍 Get Recommendations", type="primary"):
        if not product_input or not product_input.strip():
            st.warning("Please enter or select a product name.")
        else:
            matched_product, recommendations = get_recommendations(product_input, top_n=5)

            if matched_product is None:
                st.error(
                    f"❌ No product matching '{product_input}' was found in the catalogue. "
                    "Try selecting from the dropdown list instead."
                )
            else:
                if matched_product != product_input.strip().upper():
                    st.info(f"Closest match found: **{matched_product}**")

                st.success(f"✅ Top 5 Recommendations for **{matched_product}**")

                for i, (prod, score) in enumerate(recommendations, 1):
                    col_a, col_b, col_c = st.columns([1, 6, 2])
                    with col_a:
                        st.markdown(f"### #{i}")
                    with col_b:
                        st.markdown(f"**{prod}**")
                        st.progress(min(max(float(score), 0.0), 1.0))
                    with col_c:
                        st.metric("Similarity", f"{score:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTATION PAGE
# ──────────────────────────────────────────────────────────────────────────
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Predictor")
    st.markdown(
        "Enter a customer's **RFM values** to predict which segment they "
        "belong to, based on the trained KMeans clustering model."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0, max_value=1000, value=30, step=1,
            help="Number of days since the customer's most recent purchase."
        )
    with col2:
        frequency = st.number_input(
            "Frequency (number of purchases)",
            min_value=1, max_value=1000, value=5, step=1,
            help="Total number of unique purchase invoices by the customer."
        )
    with col3:
        monetary = st.number_input(
            "Monetary (total spend in £)",
            min_value=0.0, max_value=1000000.0, value=500.0, step=10.0,
            help="Total amount spent by the customer."
        )

    if st.button("🔮 Predict Segment", type="primary"):
        segment, cluster_id = predict_segment(recency, frequency, monetary)
        info = SEGMENT_INFO.get(segment, {
            "emoji": "❓", "color": "#888888",
            "description": "Unclassified segment.",
            "action": "No specific action defined."
        })

        st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color:{info['color']}22; padding: 24px;
                        border-radius: 12px; border-left: 6px solid {info['color']};">
                <h2 style="margin-top:0;">{info['emoji']} Predicted Segment: <b>{segment}</b></h2>
                <p style="font-size:16px;">{info['description']}</p>
                <p style="font-size:15px;"><b>Recommended Action:</b> {info['action']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Input Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recency", f"{recency} days")
        c2.metric("Frequency", f"{frequency} orders")
        c3.metric("Monetary", f"£{monetary:,.2f}")

        # Show where this customer falls relative to the population
        st.markdown("### How does this compare to existing customers?")
        comp_df = rfm_data.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
        comp_df.loc["Your Input"] = [recency, frequency, monetary]
        st.dataframe(comp_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Shopper Spectrum | RFM Analysis + KMeans Clustering + Collaborative Filtering")
