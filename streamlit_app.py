# -*- coding: utf-8 -*-
"""Main Streamlit application entry point for Smart Shop Assistant."""
import streamlit as st
import torch

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

# Import chat history utilities
from utils.chat_history import load_chat_history_from_supabase, save_chat_history_to_supabase
from utils.supabase_test import display_supabase_status

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

st.title("🛍️ Your Smart Shopping Assistant")
st.markdown("""
<span style='font-size:17px; color:#333;'>
Hello! 👋
Discover smarter shopping — tailored to your preferences, budget, and style.
Let's make finding the perfect product effortless and inspiring. 💫
</span>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history_from_supabase()

if "find_similar_to" in st.session_state and st.session_state.find_similar_to:
    product_id = st.session_state.pop("find_similar_to") 

    with st.spinner(f"Finding items similar to product {product_id}..."):
        similar_products = get_similar_products(product_id, qdrant_client)

        if similar_products:
            try:
                source_product_name = df_cleaned[df_cleaned['product_id'] == product_id]['product_name'].iloc[0]
                content = f"Here are some recommendations similar to '{source_product_name}':"
            except (IndexError, KeyError, ValueError):
                content = f"Here are some recommendations similar to product ID {product_id}:"
        else:
            content = f"Sorry, I couldn't find any similar items for product ID {product_id}."

        new_message = {
            "role": "assistant",
            "content": content,
            "products": similar_products
        }
        st.session_state.messages.append(new_message)
        save_chat_history_to_supabase(st.session_state.messages)
        st.rerun() # نعيد التنفيذ لعرض الرسالة الجديدة عبر حلقة العرض الرئيسية

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
                if st.button("Show more..", key=button_key):
                    st.session_state[num_visible_key] = min(num_visible + 3, total_products)
                    st.rerun()
        if message.get("image"):
            from io import BytesIO
            from PIL import Image
            img = Image.open(BytesIO(message["image"].encode("latin1")))
            st.image(img, caption="uploaded image", use_container_width=False , output_format="JPEG" , width = 100)



# --- 1. التعامل مع مدخلات المستخدم ---
if prompt := st.chat_input("What are you looking for today?", accept_file=True):
    # استخلاص النص والصورة (إن وُجدت)
    user_text = prompt.text.strip() if prompt.text else ""
    image_bytes = None

    if prompt.files:
        uploaded_file = prompt.files[0]
        # نقرأ المحتوى بايتياً بشكل صريح
        try:
            image_bytes = uploaded_file.read()
            if not isinstance(image_bytes, (bytes, bytearray)):
                image_bytes = bytes(image_bytes)
        except Exception as e:
            st.warning(f"Could not read image bytes: {e}")
            image_bytes = None
  # بايتات الصورة

    # حفظ الرسالة في الجلسة
    user_message = {
    "role": "user",
    "content": user_text or "[Image uploaded]",
    "image": image_bytes.decode("latin1") if image_bytes else None
}

    st.session_state.messages.append(user_message)

    # تمرير النص والصورة للمعالجة
    st.session_state.prompt_to_process = {"text": user_text, "image": image_bytes}

    save_chat_history_to_supabase(st.session_state.messages)
    st.rerun()


# --- 2. تنفيذ المعالجة عند وجود طلب ---
if "prompt_to_process" in st.session_state:
    data = st.session_state.pop("prompt_to_process")
    text_prompt = data["text"]
    image_data = data["image"]

    with st.chat_message("assistant"):
        with st.spinner("Your SmartShop Assistant is thinking..."):
            recommendation_text, selected_products = get_recommendation(
                text_prompt,
                st.session_state.messages,
                embedding_model,
                qdrant_client,
                gemini_model,
                image_bytes=image_data,
                clip_model=clip_model,
                clip_processor=clip_processor
            )

    assistant_message = {
        "role": "assistant",
        "content": recommendation_text,
        "products": selected_products
    }

    st.session_state.messages.append(assistant_message)
    save_chat_history_to_supabase(st.session_state.messages)
    st.rerun()
