# -*- coding: utf-8 -*-
"""Main Streamlit application entry point for Smart Shop Assistant."""
import streamlit as st
import torch

# Import configuration
from config import COLLECTION_NAME

# Import model loaders
from models.loaders import load_models_and_clients

# Import data loaders
from data.loader import load_and_prepare_data

# Import RAG components
from rag.synthesis import get_recommendation
from rag.retrieval import get_similar_products

# Import UI components
from ui.components import display_products_in_grid

# --- 1. Page and App Configuration ---
st.set_page_config(
    page_title="SmartShop Assistant",
    page_icon="🛍️",
    layout="centered"
)

# --- 2. Argument Parsing and Initialization -----
try:
    GOOGLE_API_KEY_ARG = st.secrets["GOOGLE_API_KEY"]
    QDRANT_API_KEY_ARG = st.secrets["QDRANT_API_KEY"]
except KeyError:
    st.error("FATAL ERROR: API keys not found. Please add them to the app's secrets in Streamlit Community Cloud or in a local .streamlit/secrets.toml file.")
    st.stop()

embedding_model, qdrant_client, gemini_model, clip_model, clip_processor = load_models_and_clients(GOOGLE_API_KEY_ARG, QDRANT_API_KEY_ARG)
df_cleaned = load_and_prepare_data()
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    if qdrant_client.get_collection(collection_name=COLLECTION_NAME).points_count == 0:
        st.warning("Warning: Qdrant database collection is empty. Please run population cells in notebook.")
except Exception as e:
    st.error(f"Could not connect to Qdrant collection '{COLLECTION_NAME}'. Error: {e}")
    st.stop()

# --- 3. Streamlit User Interface ---
st.title("🛍️ Your Smart Shopping Assistant")
st.markdown("""
<span style='font-size:17px; color:#333;'>
Hello! 👋
Discover smarter shopping — tailored to your preferences, budget, and style.
Let's make finding the perfect product effortless and inspiring. 💫
</span>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "find_similar_to" in st.session_state and st.session_state.find_similar_to:
    product_id = st.session_state.find_similar_to
    st.session_state.find_similar_to = None

    with st.spinner(f"Finding items similar to product {product_id}..."):
        similar_products = get_similar_products(product_id, qdrant_client)

        if similar_products:
            try:
                source_product_name = df_cleaned[df_cleaned['product_id'] == product_id]['product_name'].iloc[0]
                content = f"Here are some recommendations similar to '{source_product_name}':"
            except (IndexError, KeyError):
                content = f"Here are some recommendations similar to product ID {product_id}:"
        else:
            content = f"Sorry, I couldn't find any similar items for product ID {product_id}."

        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "products": similar_products
        })
        st.rerun()

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "products" in message and message["products"]:
            total_products = len(message["products"])

            num_visible_key = f"num_visible_{i}"

            if num_visible_key not in st.session_state:
                st.session_state[num_visible_key] = 3

            num_visible = st.session_state[num_visible_key]

            products_to_display = message["products"][:num_visible]
            display_products_in_grid(products_to_display, message_key_prefix=i)

            if num_visible < total_products:
                button_key = f"btn_more_{i}"
                if st.button("more ..", key=button_key):
                    st.session_state[num_visible_key] += 3
                    st.rerun()

if prompt := st.chat_input("What are you looking for today?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "products": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("your SmartShop Assistant is thinking..."):
            recommendation_text, selected_products = get_recommendation(
                prompt, 
                st.session_state.messages,
                embedding_model,
                qdrant_client,
                gemini_model
            )

            st.markdown(recommendation_text)

            if selected_products:
                total_products = len(selected_products)

                num_visible_key = "num_visible_current"

                if num_visible_key not in st.session_state:
                    st.session_state[num_visible_key] = 3

                if "new_prompt" not in st.session_state or st.session_state.new_prompt != prompt:
                    st.session_state.new_prompt = prompt
                    st.session_state[num_visible_key] = 3

                num_visible = st.session_state[num_visible_key]

                products_to_display = selected_products[:num_visible]
                display_products_in_grid(products_to_display, message_key_prefix="current")

                if num_visible < total_products:
                    if st.button("Show more..", key="btn_more_current"):
                        st.session_state[num_visible_key] += 3
                        st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": recommendation_text,
        "products": selected_products
    })
