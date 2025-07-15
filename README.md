# 🛒 Advanced AI-Powered E-Commerce Catalog

A mini e-commerce web app with a feature-rich product catalog, **personalized AI recommendations**, and thoughtful future integration with blockchain technology.  
Built using Python, Streamlit, and scikit-learn.

---

## 🚀 Demo

- **Filters**: Browse and filter by category, brand, color, price, or search for any feature/keyword.
- **Personalization**: Like products you’re interested in—AI learns your preferences!
- **AI Recommendations**: Get 5 explainable product recommendations based on your likes and searches.
- **Rich Product Data**: Realistic catalog with brand, color, stock, keywords, etc.
- **Blockchain Integration (Bonus)**: Conceptual extension for future-proof features.

---

## 🏗️ How to Run

1. **Clone or download** this repository.
2. **Install requirements**:
    ```bash
    pip install streamlit scikit-learn numpy
    ```
3. **Start the app**:
    ```bash
    streamlit run app.py
    ```
4. Open the local link in your browser (usually `http://localhost:8501`).

---

## 🤖 AI Feature: Hybrid Personalized Recommendations

- **Content-Based Filtering**:  
  Uses NLP (`TfidfVectorizer`) on a blend of product features (description, keywords, category, brand, color) to represent each product as a feature vector.
- **User Interest Learning**:  
  Tracks products you "like" during your session; if none, uses your current search query.
- **Cosine Similarity**:  
  Finds and ranks products closest to your interests or search as recommendations.
- **Explainable AI**:  
  Each recommendation states why it’s suggested (“Recommended based on your likes” or “Recommended based on your search”).

---

## 📝 Example Features

- **Rich filtering and sorting**:  
  Category, brand, color, price slider, and smart search bar.
- **Product details**:  
  Brand, color, stock, keywords, and more.
- **Like/reset preference buttons**:  
  User session tracks what you like, making AI smarter.
- **Explainable recommendations**:  
  Each suggested product shows a reason.
- **Blockchain integration idea**:  
  See the info section in the app for how this AI could connect with blockchain (loyalty, privacy, token-gated pricing).

---

## 🛠️ Tech Stack

- **Python 3**
- **Streamlit** (fast, interactive UI)
- **scikit-learn** (NLP and similarity calculation)
- **NumPy** (vector math)

---

## 📦 Notable Assumptions

- Sample product data is in-memory (not from a database).
- No authentication; personalization is per session.
- Product “like” events are tracked for the current session only.
- AI recommendations work with both likes and searches.
- Blockchain features are conceptual only (not implemented).

---

## 🔗 Blockchain + AI Integration (Conceptual Bonus)

> - **Token-gated pricing:**  
>   Special discounts or products unlock if your crypto wallet holds specific tokens.
> - **On-chain preferences:**  
>   User preferences and history could be stored securely and portably on a blockchain.
> - **Loyalty smart contracts:**  
>   AI-driven engagement (likes, purchases) could earn smart-contract-based loyalty points.

---

## 🎥 Video Walkthrough

See the included Loom (or screen-recorded) video for a 2–3 minute demo showing:
- Filtering, searching, and browsing products.
- “Liking” products and seeing smarter recommendations.
- Where and how the AI logic works.
- Blockchain idea explanation.

---

## 👤 Author

Sepehr (@sepehrrezaee)  
[https://github.com/SepehrRezaee]

---

**Ready to explore?**  
Just run the app and experience AI-powered, explainable recommendations—your future-ready e-commerce assistant!
