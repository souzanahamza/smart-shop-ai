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
# ==============================================================================
# 🎨 3. UI Presentation and State Initialization
# ==============================================================================

st.title("🛍️ Your Smart Shopping Assistant")
st.markdown("""
<span style='font-size:17px; color:#333;'>
Hello! 👋
Discover smarter shopping — tailored to your preferences, budget, and style.
Let's make finding the perfect product effortless and inspiring. 💫
</span>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
# Load chat history from Supabase only once at the beginning of the session.
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history_from_supabase()

# ==============================================================================
# 💬 4. Chat History Display (Single Source of Truth)
# ==============================================================================
# This loop renders the entire chat history from the session state.
# It runs on every script rerun, ensuring the UI is always up-to-date.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message contains products, display them in a grid.
        if message.get("products"):
            total_products = len(message["products"])
            num_visible_key = f"num_visible_{i}"
            if num_visible_key not in st.session_state:
                st.session_state[num_visible_key] = 3  # Initially show 3 products
            
            num_visible = st.session_state[num_visible_key]
            products_to_display = message["products"][:num_visible]
            
            display_products_in_grid(products_to_display, message_key_prefix=i)
            
            # "Show more" button for product pagination
            if num_visible < total_products:
                if st.button("Show more..", key=f"btn_more_{i}"):
                    st.session_state[num_visible_key] = min(num_visible + 3, total_products)
                    st.rerun()

# ==============================================================================
# 🧠 5. AI Response Generation (Asynchronous-like Logic)
# ==============================================================================
# This section handles the "thinking" part of the assistant. It runs only when
# specific flags are set in the session state.

# --- 5a. Generate "More like this" Response ---
if st.session_state.get("find_similar_to"):
    product_id = st.session_state.get("find_similar_to")
    assistant_message = None

    try:
        # Display the spinner inside the chat bubble for a seamless UX
        with st.chat_message("assistant"):
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
                
                assistant_message = {"role": "assistant", "content": content, "products": similar_products or []}

    except Exception as e:
        st.error("Sorry, I couldn't process your request right now. Please try again.")
        print(f"Error during get_similar_products: {e}") # For debugging
        assistant_message = {"role": "assistant", "content": "Sorry, an unexpected error occurred."}
    
    finally:
        # This block always runs, ensuring the state is cleaned up
        if assistant_message:
            st.session_state.messages.append(assistant_message)
            try: save_chat_history_to_supabase(st.session_state.messages)
            except Exception: pass # Don't block UI if saving fails
        
        del st.session_state["find_similar_to"] # Clean up the flag
        st.rerun()

# --- 5b. Generate Response to a New Prompt ---
if st.session_state.get("generating_response"):
    prompt_to_process = st.session_state.get("last_prompt")
    assistant_message = None

    try:
        with st.chat_message("assistant"):
            with st.spinner("your SmartShop Assistant is thinking..."):
                recommendation_text, selected_products = get_recommendation(
                    prompt_to_process, st.session_state.messages, embedding_model, qdrant_client, gemini_model
                )
        assistant_message = {"role": "assistant", "content": recommendation_text, "products": selected_products}
    
    except Exception as e:
        st.error("I'm sorry, I encountered an issue. Please try asking in a different way.")
        print(f"Error during get_recommendation: {e}") # For debugging
        assistant_message = {"role": "assistant", "content": "I'm having trouble connecting right now. Please try again later."}

    finally:
        if assistant_message:
            st.session_state.messages.append(assistant_message)
            try: save_chat_history_to_supabase(st.session_state.messages)
            except Exception: pass
        
        # Clean up state flags to prevent re-running
        del st.session_state["generating_response"]
        if "last_prompt" in st.session_state:
            del st.session_state["last_prompt"]
        
        st.rerun()

# ==============================================================================
# ⌨️ 6. User Input Handling
# ==============================================================================
# This is the entry point for user interaction. It only handles the first step:
# capturing the input and setting a flag for the generation logic.
if prompt := st.chat_input(
    "What are you looking for today?",
    # Disable the input while the assistant is busy with any task
    disabled=st.session_state.get("generating_response", False) or st.session_state.get("find_similar_to", False)
):
    # 1. Immediately add the user's message to the state to display it
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # 2. Set flags to trigger the AI response generation on the next rerun
    st.session_state.last_prompt = prompt
    st.session_state.generating_response = True
    
    # 3. Save the updated chat history securely
    try:
        save_chat_history_to_supabase(st.session_state.messages)
    except Exception:
        pass # Don't block UI
    
    # 4. Rerun the app immediately. This makes the user's message appear instantly
    # and starts the "thinking" process defined in section 5.
    st.rerun()