import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load products from CSV
@st.cache_data
def load_products(path="./Products.csv"):
    df = pd.read_csv(path)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    # Ensure all filter columns are strings with no NaNs
    for col in ["category", "brand", "color", "desc", "keywords"]:
        df[col] = df[col].fillna("N/A").astype(str)
    return df

products_df = load_products()
sample_products = products_df.to_dict(orient="records")

# 2. Build TF-IDF matrix
docs = [
    f"{p['desc']} {p['keywords']} {p['category']} {p['brand']} {p['color']}"
    for p in sample_products
]
vectorizer = TfidfVectorizer(stop_words="english").fit(docs)
product_matrix = vectorizer.transform(docs).toarray()

# 3. Streamlit UI Setup
st.set_page_config(layout="wide", page_title="🛒 AI-Powered Catalog")
st.title("🛒 Product Catalog with Advanced AI Recommendations")

# Session state
if "liked" not in st.session_state: st.session_state.liked = []
if "disliked" not in st.session_state: st.session_state.disliked = []

# 4. Filters & Search Controls
cols = st.columns(5)
category = cols[0].selectbox("Category", ["All"] + sorted(products_df["category"].unique()))
brand = cols[1].selectbox("Brand", ["All"] + sorted(products_df["brand"].unique()))
color = cols[2].selectbox("Color", ["All"] + sorted(products_df["color"].unique()))
price = cols[3].slider(
    "Max Price",
    float(products_df["price"].min()), float(products_df["price"].max()),
    float(products_df["price"].max())
)
sort_by = cols[4].selectbox("Sort By", ["Rating", "Price (Low→High)", "Price (High→Low)", "Stock"])
search = st.text_input("🔎 Search name, desc, or keywords")

# 5. Apply filtering to DataFrame
def filter_df(df):
    df_f = df.copy()
    if category != "All":
        df_f = df_f[df_f["category"] == category]
    if brand != "All":
        df_f = df_f[df_f["brand"] == brand]
    if color != "All":
        df_f = df_f[df_f["color"] == color]
    df_f = df_f[df_f["price"] <= price]
    if search:
        q = search.lower()
        df_f = df_f[df_f.apply(lambda r: q in r["name"].lower() or q in r["desc"].lower() or q in r["keywords"].lower(), axis=1)]
    if sort_by == "Rating":
        df_f = df_f.sort_values("rating", ascending=False)
    elif sort_by == "Price (Low→High)":
        df_f = df_f.sort_values("price", ascending=True)
    elif sort_by == "Price (High→Low)":
        df_f = df_f.sort_values("price", ascending=False)
    else:
        df_f = df_f.sort_values("stock", ascending=False)
    return df_f

filtered_df = filter_df(products_df)

# 6. Display Scrollable Catalog
# 6. Display Top 5 Search-Related Products if search is entered
if search:
    search_vec = vectorizer.transform([search]).toarray()
    similarities = cosine_similarity(search_vec, product_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:5]
    st.subheader("🔍 Top 5 Products Related to Your Search")
    for idx in top_indices:
        p = sample_products[idx]
        st.markdown(f"**{p['name']}** — ${p['price']} | ⭐{p['rating']} | Stock: {p['stock']}")
        st.caption(f"{p['desc']}")
        st.write("---")
else:
    # 6. Display Scrollable Catalog if no search
    st.subheader("📦 Product Catalog")
    st.dataframe(filtered_df, height=400)


# 7. Like/Dislike Buttons
# st.subheader("👍 👎 Feedback")
# for idx, row in filtered_df.reset_index().iterrows():
#     col1, col2, _ = st.columns([1,1,8])
#     with col1:
#         if st.button("👍", key=f"like_{row['index']}"):
#             st.session_state.liked.append((int(row["index"]), time.time()))
#     with col2:
#         if st.button("👎", key=f"dislike_{row['index']}"):
#             st.session_state.disliked.append(int(row["index"]))

st.markdown("---")

# 8. AI Recommendation Logic
def compute_affinities():
    cat_cnt, brand_cnt = {}, {}
    now = time.time()
    for idx, ts in st.session_state.liked:
        decay = max(0.1, 1.0 - (now - ts)/86400)
        p = sample_products[idx]
        cat_cnt[p["category"]] = cat_cnt.get(p["category"], 0) + decay
        brand_cnt[p["brand"]] = brand_cnt.get(p["brand"], 0) + decay
    total_c = sum(cat_cnt.values()) or 1
    total_b = sum(brand_cnt.values()) or 1
    return {c: v/total_c for c, v in cat_cnt.items()}, {b: v/total_b for b, v in brand_cnt.items()}

def recommendation_scores():
    liked_idxs = [i for i,_ in st.session_state.liked]
    user_vec = np.mean(product_matrix[liked_idxs], axis=0) if liked_idxs else np.mean(product_matrix, axis=0)
    sims = cosine_similarity([user_vec], product_matrix).flatten()

    cat_aff, brand_aff = compute_affinities()
    cat_score = np.array([cat_aff.get(p["category"], 0) for p in sample_products])
    brand_score = np.array([brand_aff.get(p["brand"], 0) for p in sample_products])
    ratings = np.array([p["rating"] for p in sample_products])
    norm_rating = (ratings - ratings.min())/(ratings.max() - ratings.min() + 1e-9)

    α = 0.5 + 0.1 * (len(st.session_state.liked) - len(st.session_state.disliked))
    β, γ = 0.2, 0.1
    δ = 1 - (α + β + γ)
    composite = α * sims + β * cat_score + γ * brand_score + δ * norm_rating

    for d in st.session_state.disliked:
        composite[d] = -np.inf
        neighbors = cosine_similarity([product_matrix[d]], product_matrix).flatten()
        composite -= 0.3 * neighbors

    ranked, final, seen = np.argsort(composite)[::-1], [], set()
    for idx in ranked:
        cat = sample_products[idx]["category"]
        if len(final)<2 or cat not in seen:
            final.append(idx); seen.add(cat)
        if len(final)>=5:
            break
    return final

recs = recommendation_scores()

# 9. Show Recommendations
st.header("🤖 AI-Enhanced Personalized Recommendations")
cat_aff, brand_aff = compute_affinities()
for idx in recs:
    p = sample_products[idx]
    st.markdown(f"**{p['name']}** — ${p['price']} | ⭐{p['rating']} | Stock: {p['stock']}")
    st.caption(f"Category affinity={cat_aff.get(p['category'],0):.2f}, Brand affinity={brand_aff.get(p['brand'],0):.2f}")

# 10. Blockchain Integration (Bonus)
with st.expander("💡 AI + Blockchain Integration Ideas"):
    st.markdown("""
    - **Token-gated pricing:** unlock discounts for token holders.
    - **On-chain preferences:** store user prefs securely on blockchain.
    - **Loyalty smart contracts:** earn & redeem loyalty points transparently.
    """)

# 11. Reset Controls
if st.button("🔄 Reset Feedback"):
    st.session_state.liked.clear()
    st.session_state.disliked.clear()
