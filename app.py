import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Advanced, Feature-Rich Product List ----------
sample_products = [
    {
        "name": "Laptop Pro 15",
        "brand": "TechBrand",
        "price": 1499, "category": "Electronics", "color": "Silver",
        "desc": "15-inch high-performance laptop for creators and professionals.",
        "rating": 4.7, "stock": 12, "keywords": "laptop, professional, high-end"
    },
    {
        "name": "Smartphone X12",
        "brand": "PhoneCo",
        "price": 999, "category": "Electronics", "color": "Black",
        "desc": "Flagship smartphone with AI camera and OLED display.",
        "rating": 4.6, "stock": 18, "keywords": "smartphone, camera, OLED"
    },
    {
        "name": "Noise Cancel Headphones",
        "brand": "SoundMax",
        "price": 299, "category": "Electronics", "color": "White",
        "desc": "Wireless over-ear headphones with top-tier noise cancellation.",
        "rating": 4.5, "stock": 25, "keywords": "headphones, wireless, music, noise cancellation"
    },
    {
        "name": "Smartwatch Fit",
        "brand": "PhoneCo",
        "price": 249, "category": "Electronics", "color": "Black",
        "desc": "Fitness-focused smartwatch with heart rate and sleep tracking.",
        "rating": 4.2, "stock": 30, "keywords": "smartwatch, fitness, tracker"
    },
    {
        "name": "Eco T-Shirt",
        "brand": "GreenWear",
        "price": 39, "category": "Clothing", "color": "Green",
        "desc": "Organic cotton t-shirt, eco-friendly and stylish.",
        "rating": 4.1, "stock": 40, "keywords": "t-shirt, organic, cotton, sustainable"
    },
    {
        "name": "Urban Jeans",
        "brand": "DenimCo",
        "price": 89, "category": "Clothing", "color": "Blue",
        "desc": "Slim-fit urban jeans for modern comfort and style.",
        "rating": 4.3, "stock": 22, "keywords": "jeans, denim, slim-fit, fashion"
    },
    {
        "name": "Running Sneakers",
        "brand": "Sportify",
        "price": 120, "category": "Footwear", "color": "Red",
        "desc": "Lightweight running sneakers with maximum grip.",
        "rating": 4.6, "stock": 16, "keywords": "sneakers, running, sport"
    },
    {
        "name": "Classic Leather Shoes",
        "brand": "Elegance",
        "price": 170, "category": "Footwear", "color": "Brown",
        "desc": "Classic leather shoes for formal and business wear.",
        "rating": 4.4, "stock": 15, "keywords": "leather, shoes, formal, business"
    },
    {
        "name": "Backpack Explorer",
        "brand": "AdventureGear",
        "price": 70, "category": "Accessories", "color": "Navy",
        "desc": "Multi-pocket, water-resistant backpack for city and hiking.",
        "rating": 4.5, "stock": 20, "keywords": "backpack, hiking, travel, waterproof"
    },
    {
        "name": "Classic Coffee Maker",
        "brand": "HomeBrew",
        "price": 110, "category": "Home Appliances", "color": "White",
        "desc": "Easy-to-use coffee maker with programmable timer.",
        "rating": 4.2, "stock": 18, "keywords": "coffee maker, appliance, kitchen"
    },
    {
        "name": "Blender UltraMix",
        "brand": "KitchenPro",
        "price": 140, "category": "Home Appliances", "color": "Black",
        "desc": "Powerful blender for smoothies, soups, and more.",
        "rating": 4.6, "stock": 10, "keywords": "blender, kitchen, appliance, smoothies"
    },
    {
        "name": "Bestseller Novel: The Journey",
        "brand": "BookWorld",
        "price": 23, "category": "Books", "color": "N/A",
        "desc": "Award-winning adventure novel with thrilling story.",
        "rating": 4.9, "stock": 50, "keywords": "book, novel, adventure, bestseller"
    },
    {
        "name": "Desk Lamp SmartLED",
        "brand": "BrightTech",
        "price": 58, "category": "Home Appliances", "color": "Gray",
        "desc": "Smart LED desk lamp with color modes and USB charging.",
        "rating": 4.4, "stock": 28, "keywords": "lamp, desk, LED, smart"
    },
    {
        "name": "Leather Wallet",
        "brand": "Elegance",
        "price": 45, "category": "Accessories", "color": "Black",
        "desc": "Handmade genuine leather wallet with RFID protection.",
        "rating": 4.7, "stock": 32, "keywords": "wallet, leather, RFID"
    },
    {
        "name": "Cookbook: World Flavors",
        "brand": "BookWorld",
        "price": 29, "category": "Books", "color": "N/A",
        "desc": "Delicious global recipes for home cooks.",
        "rating": 4.6, "stock": 40, "keywords": "cookbook, recipes, world, cuisine"
    }
]

# ---------- Build Feature/Content Vectors for AI Recommendation ----------
product_texts = [
    f"{p['desc']} {p['keywords']} {p['category']} {p['brand']} {p['color']}"
    for p in sample_products
]
vectorizer = TfidfVectorizer(stop_words="english").fit(product_texts)
product_matrix = vectorizer.transform(product_texts)

# ---------- Streamlit UI -------------
st.set_page_config(page_title="🛒 AI-Powered E-Commerce Demo", layout="wide")
st.title("🛒 E-Commerce Product Catalog with Advanced AI Recommendation")

# --- User Filter Controls ---
cols = st.columns(5)
category = cols[0].selectbox("Category", ["All"] + sorted({p["category"] for p in sample_products}))
brand = cols[1].selectbox("Brand", ["All"] + sorted({p["brand"] for p in sample_products}))
color = cols[2].selectbox("Color", ["All"] + sorted({p["color"] for p in sample_products if p["color"] != "N/A"]))
price = cols[3].slider("Max Price", min_value=10, max_value=2000, value=2000, step=10)
sort_by = cols[4].selectbox("Sort By", ["Rating", "Price (Low to High)", "Price (High to Low)", "Stock"])

search = st.text_input("🔎 Search for product, feature, or keyword")

# --- Filter Products Function ---
def filter_products(products):
    result = products
    if category != "All":
        result = [p for p in result if p["category"] == category]
    if brand != "All":
        result = [p for p in result if p["brand"] == brand]
    if color != "All":
        result = [p for p in result if p["color"] == color]
    result = [p for p in result if p["price"] <= price]
    if search:
        q = search.lower()
        result = [p for p in result if q in p["name"].lower() or q in p["desc"].lower() or q in p["keywords"].lower()]
    if sort_by == "Rating":
        result = sorted(result, key=lambda x: -x["rating"])
    elif sort_by == "Price (Low to High)":
        result = sorted(result, key=lambda x: x["price"])
    elif sort_by == "Price (High to Low)":
        result = sorted(result, key=lambda x: -x["price"])
    elif sort_by == "Stock":
        result = sorted(result, key=lambda x: -x["stock"])
    return result

filtered_products = filter_products(sample_products)

# --- Product Display ---
st.subheader("📦 Catalog")
for p in filtered_products:
    with st.expander(f"{p['name']} | {p['brand']} | ${p['price']} | ⭐{p['rating']}"):
        st.write(
            f"**Description:** {p['desc']}\n\n"
            f"**Category:** {p['category']}\n\n"
            f"**Brand:** {p['brand']}\n\n"
            f"**Color:** {p['color']}\n\n"
            f"**Stock:** {p['stock']} units\n\n"
            f"**Keywords:** {p['keywords']}"
        )
        # --- User Interest Tracking ---
        if st.button(f"👍 Interested in {p['name']}", key=f"like_{p['name']}"):
            if "liked" not in st.session_state:
                st.session_state.liked = set()
            st.session_state.liked.add(p["name"])

st.write("---")

# --------- ADVANCED AI RECOMMENDATION SYSTEM --------------
st.header("🤖 AI Recommendations (Personalized & Explainable)")

# --- Build User Profile Vector (from interests or search) ---
def get_user_interest_vector():
    liked_names = st.session_state.get("liked", set())
    liked_indices = [i for i, p in enumerate(sample_products) if p["name"] in liked_names]
    if liked_indices:
        # User has liked some products: use average of their vectors
        user_vec = np.mean(product_matrix[liked_indices].toarray(), axis=0)
        reason = "based on products you expressed interest in"
    elif search:
        user_vec = vectorizer.transform([search]).toarray()[0]
        reason = "based on your search"
    else:
        user_vec = np.mean(product_matrix.toarray(), axis=0)
        reason = "top picks for you"
    return user_vec.reshape(1, -1), reason

user_vector, explain_reason = get_user_interest_vector()

# --- Compute Cosine Similarity for Recommendations ---
sims = cosine_similarity(user_vector, product_matrix).flatten()
top_idx = sims.argsort()[::-1]
# Don't show products the user already liked/interacted with in recommendations
liked_names = st.session_state.get("liked", set())
recommendations = [sample_products[i] for i in top_idx if sample_products[i]["name"] not in liked_names][:5]

if recommendations:
    for p in recommendations:
        st.markdown(
            f"**{p['name']}**  \n"
            f"Category: {p['category']} | Brand: {p['brand']} | Color: {p['color']}  \n"
            f"Price: ${p['price']} | ⭐ {p['rating']}  \n"
            f"_{p['desc']}_"
        )
        st.caption(f"→ Recommended {explain_reason}")
        st.write("---")
else:
    st.write("No AI recommendations available (like more products or refine your search).")

# ---- Optional: Reset User Preferences ----
if st.button("🔄 Reset My Preferences"):
    st.session_state.liked = set()

# --- Blockchain Integration Example ---
with st.expander("💡 How could AI + Blockchain work here?"):
    st.markdown("""
    - **Token-gated pricing**: Certain products or discounts are unlocked if your wallet holds a special token.
    - **On-chain preferences**: Your preferences and purchase history are stored securely on the blockchain, making recommendations more portable and privacy-friendly.
    - **Loyalty smart contracts**: You earn blockchain-based loyalty points with each purchase or product interaction, redeemable for exclusive offers.
    """)

st.info("Demo app by Sepehr — for technical interviews and prototyping.")
