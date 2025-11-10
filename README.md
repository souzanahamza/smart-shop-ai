# SmartShop Assistant 🛍️

An intelligent, AI-powered shopping assistant built with Streamlit and a Retrieval-Augmented Generation (RAG) pipeline to provide a seamless, multimodal (text + image) product discovery experience.

---

## ✨ About The Project

Finding the perfect product in a large e-commerce catalog can be challenging.  
SmartShop Assistant solves this by allowing users to **describe or show** what they’re looking for — through text, image, or both.  

The assistant understands natural language queries, interprets uploaded images using CLIP embeddings, and fuses both modalities (text + image) to deliver precise, visually aligned product recommendations.  
A powerful Large Language Model (Google Gemini) then analyzes the retrieved results and generates friendly, context-aware responses.

---

## 🚀 Features

- **Conversational Search:** Chat naturally with the AI to find products like you would with a real sales assistant.  
- **Context-Aware Recommendations:** The assistant remembers the conversation to refine suggestions dynamically.  
- **Semantic Product Search:** Powered by `sentence-transformers` and **Qdrant**, enabling retrieval by meaning, not just keywords.  
- **Visual & Hybrid Search:** Upload an image (optionally with text like “I want this but in blue”) to find visually and semantically similar items using CLIP embeddings.  
- **“More Like This” Visual Matching:** Instantly explore products that look similar to any item in the catalog.  
- **Intelligent Synthesis:** Uses **Google Gemini** to generate human-like shopping advice based on search results.  
- **Cloud Logging & Memory:** Chat interactions and recommendations are stored securely in **Supabase** for history and insights.  
- **Modular & Scalable:** Clean architecture built for easy maintenance and future expansion.  

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Vector Database:** [Qdrant](https://qdrant.tech/)  
- **LLM & Generative AI:** [Google Gemini](https://ai.google.dev/)  
- **Embeddings:**
  - [SentenceTransformers (e5-base-v2)](https://huggingface.co/intfloat/e5-base-v2) – for text queries  
  - [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) – for image and hybrid understanding  
- **Multimodal Fusion:** Weighted CLIP-based hybrid retrieval (text + image)  
- **Cloud Storage:** [Supabase](https://supabase.com/) for chat history and logs  
- **Core Libraries:** Pandas, Transformers, PyTorch  

---

## 📂 Project Structure

````

.
├── .streamlit/
│   └── secrets.toml         # For storing API keys securely
├── data/
│   ├── combined_products2.csv
│   └── loader.py            # Logic for loading and preparing data
├── models/
│   └── loaders.py           # Loads ML models (SentenceTransformer, CLIP, Gemini)
├── rag/
│   ├── query_rewriter.py    # Rewrites & translates queries (supports hybrid text+image)
│   ├── retrieval.py         # Handles text, visual, and hybrid CLIP-based retrieval
│   └── synthesis.py         # Generates Gemini responses aware of visual context
├── ui/
│   └── components.py        # Streamlit UI helpers (product grid, buttons, etc.)
├── utils/
│   ├── helpers.py           # General helper utilities
│   ├── logging.py           # Interaction logging (Supabase)
│   └── supabase_client.py   # Database connection utilities
│
├── config.py                # Configuration for models, database, and constants
├── requirements.txt         # Project dependencies
└── streamlit_app.py         # Main entry point for the Streamlit app

````

---

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/souzanahamza/smart-shop.git
   cd smart-shop
    ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add API keys:**
   Create `.streamlit/secrets.toml`:

   ```toml
   GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
   QDRANT_API_KEY = "YOUR_QDRANT_API_KEY_HERE"
   ```

### Run the App

```bash
streamlit run streamlit_app.py
```

Then open the displayed local URL in your browser.

---

## 🧠 How It Works

1. **User Interaction:** The user enters a query — text, image, or both.
2. **Query Optimization:** The system rewrites and translates vague or multilingual input into a precise English query.
3. **Retrieval:** The system searches Qdrant using text embeddings, image embeddings, or a hybrid CLIP vector.
4. **Synthesis:** Gemini reviews retrieved results and generates a friendly, contextual recommendation message.
5. **Display:** Products are shown in an interactive Streamlit chat interface, allowing further refinements like “More like this 🔁”.

---

## 👤 Contact

**Souzana Hamza**
[GitHub Profile](https://github.com/souzanahamza)
