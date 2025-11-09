# SmartShop Assistant 🛍️

An intelligent, AI-powered shopping assistant built with Streamlit and a Retrieval-Augmented Generation (RAG) pipeline to provide a seamless and conversational product discovery experience.

## ✨ About The Project

Finding the perfect product in a large e-commerce catalog can be challenging. SmartShop Assistant solves this by allowing users to describe what they're looking for in natural language. The application understands the user's intent, searches a vector database for relevant products, and uses a powerful Large Language Model (Google's Gemini) to present smart, context-aware recommendations.

---

## 🚀 Features

-   **Conversational Search:** Interact with the assistant just like you would with a human sales associate.
-   **Context-Aware Recommendations:** The assistant remembers the conversation history to refine its suggestions.
-   **Semantic Product Search:** Powered by `sentence-transformers` and a **Qdrant** vector database to find products based on meaning, not just keywords.
-   **Intelligent Synthesis:** Uses **Google Gemini** to analyze search results and generate helpful, human-like responses.
-   **Modular & Scalable:** The codebase is organized into logical modules for easy maintenance and future expansion.

---

## 🛠️ Tech Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Vector Database:** [Qdrant](https://qdrant.tech/)
-   **LLM & Generative AI:** [Google Gemini](https://ai.google.dev/)
-   **Embedding Models:** [SentenceTransformers (e5-base-v2)](https://huggingface.co/intfloat/e5-base-v2), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
-   **Core Libraries:** Pandas, Transformers, PyTorch

---

## 📂 Project Structure

The project is structured using a modular approach to separate concerns, making the code clean and maintainable.

```
.
├── .streamlit/
│   └── secrets.toml         # For storing API keys securely
├── data/
│   ├── combined_products2.csv
│   └── loader.py            # Logic for loading and preparing data
├── models/
│   └── loaders.py           # Logic for loading ML models (embeddings, CLIP)
├── rag/
│   ├── query_rewriter.py    # Rewrites user queries for better search
│   ├── retrieval.py         # Handles searching the Qdrant vector DB
│   └── synthesis.py         # Generates the final AI response
├── ui/
│   └── components.py        # Streamlit UI helper functions (e.g., product grid)
├── utils/
│   ├── helpers.py           # General helper functions
│   └── logging.py           # Functions for logging interactions
│
├── config.py                # Main configuration file for paths and settings
├── requirements.txt         # Project dependencies
└── streamlit_app.py         # The main entry point for the Streamlit app
```

---

## 🏁 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/souzanahamza/smart-shop.git
    cd smart-shop
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Keys:**
    -   Create a folder named `.streamlit` in the project's root directory.
    -   Inside it, create a file named `secrets.toml`.
    -   Add your API keys to this file as shown below:
        ```toml
        # .streamlit/secrets.toml

        GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
        QDRANT_API_KEY = "YOUR_QDRANT_API_KEY_HERE"
        ```

### Running the App

Once the installation is complete, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

The application should now be running and accessible in your web browser!

---

## 👤 Contact

Souzana Hamza - [GitHub Profile](https://github.com/souzanahamza)
