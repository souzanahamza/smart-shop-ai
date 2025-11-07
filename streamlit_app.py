
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import google.generativeai as genai
import re
import time
import sys
import csv
from datetime import datetime
import hashlib

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import uuid


# --- 1. Page and App Configuration ---
st.set_page_config(
    page_title="SmartShop Assistant",
    page_icon="🛍️",
    layout="centered"
)

# --- 2. Caching for Performance ---
@st.cache_resource
def load_models_and_clients(google_api_key, qdrant_api_key):
    """Loads and caches all the necessary models and API clients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text Embedding Model
    embedding_model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    # Image Embedding Model (CLIP)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Qdrant Client
    qdrant_client = QdrantClient(
        url="https://c368450d-6521-40d8-9503-df090aad1775.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=qdrant_api_key
    )

    # Generative Model
    genai.configure(api_key=google_api_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest')

    return embedding_model, qdrant_client, gemini_model, clip_model, clip_processor

@st.cache_data
def load_and_prepare_data(file_path):
    """Loads and prepares the product data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file not found at {file_path}. Please check the path in your Colab environment.")
        st.stop()
    def clean_text(text): return str(text).replace("âœ“", "").strip() if pd.notna(text) else ""
    def parse_attributes(attributes_str):
        if not isinstance(attributes_str, str) or attributes_str == 'null': return ""
        try: return ", ".join([f"{attr.get('name', '')}: {attr.get('value', '')}" for attr in json.loads(attributes_str)])
        except: return ""
    def combine_product_info(row):
        parts = [f"Product Name: {clean_text(row['product_name'])}", f"Category: {clean_text(row['category'])}"]
        if clean_text(row['brand']): parts.append(f"Brand: {clean_text(row['brand'])}")
        if clean_text(row['color']): parts.append(f"Color: {clean_text(row['color'])}")
        if clean_text(row['description']): parts.append(f"Description: {clean_text(row['description'])}")
        other_attrs = parse_attributes(row['other_attributes'])
        if other_attrs: parts.append(f"Features: {other_attrs}")
        return ". ".join(parts)
    required_columns = ['product_id', 'product_name', 'description', 'category', 'brand', 'color', 'other_attributes', 'main_image']
    df[required_columns] = df[required_columns].fillna("")
    df['combined_info'] = df.apply(combine_product_info, axis=1)
    df_cleaned = df[required_columns + ['combined_info']].copy()
    df_cleaned['product_id'] = pd.to_numeric(df_cleaned['product_id'], errors='coerce')
    df_cleaned.dropna(subset=['product_id', 'combined_info', 'main_image'], inplace=True)
    df_cleaned = df_cleaned[df_cleaned['combined_info'].str.strip() != '']
    df_cleaned = df_cleaned[df_cleaned['main_image'].str.strip() != '']
    df_cleaned['product_id'] = df_cleaned['product_id'].astype('Int64')
    return df_cleaned

# --- 3. Argument Parsing and Initialization ---

try:
    GOOGLE_API_KEY_ARG = st.secrets["GOOGLE_API_KEY"]
    QDRANT_API_KEY_ARG = st.secrets["QDRANT_API_KEY"]
except KeyError:
    st.error("FATAL ERROR: API keys not found. Please add them to the app's secrets in Streamlit Community Cloud or in a local .streamlit/secrets.toml file.")
    st.stop()

FILE_PATH = 'data/combined_products2.csv'
COLLECTION_NAME = "products_e5_shein_amz_collection"
LOG_FILE = "interaction_log.csv"
# embedding_model, embedding_dim, qdrant_client, gemini_model = load_models_and_clients(GOOGLE_API_KEY_ARG, QDRANT_API_KEY_ARG)

embedding_model, qdrant_client, gemini_model, clip_model, clip_processor = load_models_and_clients(GOOGLE_API_KEY_ARG, QDRANT_API_KEY_ARG)
df_cleaned = load_and_prepare_data(FILE_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"


try:
    if qdrant_client.get_collection(collection_name=COLLECTION_NAME).points_count == 0:
        st.warning("Warning: Qdrant database collection is empty. Please run population cells in notebook.")
except Exception as e:
    st.error(f"Could not connect to Qdrant collection '{COLLECTION_NAME}'. Error: {e}")
    st.stop()

# --- 4. Core RAG Logic with Query Rewriting and Logging ---

def create_uuid_from_string(text: str) -> str:
    """Creates a consistent UUID from a string product_id."""
    NAMESPACE_DNS = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(NAMESPACE_DNS, str(text)))


def log_interaction_to_csv(log_data):
    """Appends a dictionary of log data to a CSV file."""
    file_exists = False
    try:
        with open(LOG_FILE, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

def retrieve_from_qdrant(query, top_k=20):
    if not query: return []
    try:
        query_with_prefix = "query: " + query
        query_embedding = embedding_model.encode(query_with_prefix, convert_to_numpy=True)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name="text",
                vector=query_embedding.tolist()
            ),
            limit=top_k,
            with_payload=True
        )
        # تحويل النتائج إلى كائنات ScoredPoint لتسهيل التعامل معها
        return [models.ScoredPoint.model_validate(res) for res in search_results]
    except Exception as e:
        st.error(f"Error during text retrieval: {e}")
        return []

def format_chat_history(messages):
    history_str = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Souza"
        history_str += f"{role}: {msg['content']}\n"
    return history_str.strip()

def generate_query_rewriter_prompt(user_query, chat_history):
    history_str = format_chat_history(chat_history)
    return f"""You are an expert search query optimization assistant. Your task is to rewrite a user's latest message into an optimal, concise search query for a vector database. Analyze the conversation history and the user's latest input to understand their true intent.
### CONVERSATION HISTORY
{history_str}
### LATEST USER INPUT
"{user_query}"
### TASK
Based on the full conversation, rewrite the latest user input into a self-contained search query. The query should be short, specific, and contain key terms. If the user's input is not a shopping query (e.g., "hello"), just return their original input.
- **CRITICAL INSTRUCTION: The final rewritten query MUST ALWAYS be in ENGLISH.** If the user's input is in another language, translate the core meaning into an English search query.
**Example 1:**
- History: "User: I need a gift for my father."
- Latest Input: "He likes waterproof watches."
- Rewritten Query: "waterproof watch for men"
**Example 2:**
- History: "User: I want a summer dress."
- Latest Input: "how about in red?"
- Rewritten Query: "red summer dress"
**Output only the rewritten query text, with no extra explanation or quotation marks.**
Rewritten Query:"""

def generate_optimized_search_query(user_query, chat_history):
    prompt = generate_query_rewriter_prompt(user_query, chat_history)
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip().replace('"', '').replace("'", "")
    except Exception as e:
        print(f"Error during query optimization: {e}. Falling back to original query.")
        return user_query

def generate_synthesis_prompt(user_query, retrieved_payloads, chat_history):
    history_str = format_chat_history(chat_history)
    context_str = "\n\n".join([f"ID: {p.payload.get('product_id', 'N/A')}\nName: {p.payload.get('product_name', 'N/A')}" for p in retrieved_payloads])
    # context_str = "\n\n".join([f"--- Product ---\nID: {p.get('product_id', 'N/A')}\nName: {p.get('product_name', 'N/A')}\nDetails: {p.get('combined_info', 'N/A')}" for p in retrieved_payloads])
    return f"""You are **Souza** SmartShop Assistant, an expert AI shopping assistant.
---
### 📜 CONVERSATION HISTORY
{history_str}
### 🧠 CURRENT USER INPUT
User: "{user_query}"
### 🛍️ PRODUCT CONTEXT TO ANALYZE (from a database search based on the current input)
{context_str}
---
### 🧩 INSTRUCTIONS
1.  **Analyze the CURRENT USER INPUT, using the CONVERSATION HISTORY for context.**
2.  **Determine query type:** Is it a **SHOPPING QUERY** or **OFF-TOPIC**?
3.  **If OFF-TOPIC:** Politely respond that you are a shopping assistant. Do not recommend products. Set `"selected_product_ids": []`.
4.  **If SHOPPING QUERY:** Review the PRODUCT CONTEXT carefully and choose **only the products that clearly match the user's request.**
    Exclude any products that are off-topic, irrelevant, or do not fit the user's described needs.
    Then write a friendly, helpful recommendation message explaining why these items were chosen.

5.  **PRODUCT ORDER RULE:** You may reorder or exclude products freely.
    Only include the most relevant ones in `selected_product_ids` — do not keep unrelated products.

6. Only the **first 3 selected products** will be shown to the user initially.
Write the `"recommendation_text"` so that it refers **only to these first three items**.
Your message should feel complete and natural based on the first 3 products.

7.  **Handling user intent and irrelevant results (IMPORTANT):**
    - **If the user's latest message clearly expresses what they want** (for example: "I want lipsticks", "show me red shoes", "looking for a black dress"),
      immediately recommend products from the PRODUCT CONTEXT that match — **do not ask follow-up or clarifying questions**.
    - **If the message is vague or unclear** (for example: "I need something nice" or "any suggestions?"),
      then politely ask **one short clarifying question only**.
    - **If the retrieved PRODUCT CONTEXT is irrelevant**, do not show irrelevant products; instead, offer **a general helpful suggestion** related to the query.
    - **Example Scenario:** User asks for a "gift for my husband," and you find women's jewelry. Your **Correct** Response is: "That's a thoughtful idea! To help me find the perfect gift, could you tell me a bit about his hobbies or what he likes?"
8.  **Rules:** Base your reasoning **only** on the PRODUCT CONTEXT. Output **only one valid JSON object**.
---
### ✅ RESPONSE FORMAT (JSON ONLY)
{{
  "recommendation_text": "Your friendly, context-aware response here.",
  "selected_product_ids": ["AMZN_B09CKDT4RR", "12345"]
}}"""

# ==================== النسخة الكاملة والمعدلة لهذه الدالة ====================

def get_recommendation(user_query: str, chat_history: list):
    """
    Orchestrates the recommendation process: gets a query, retrieves products,
    and uses an LLM to generate a final recommendation text and product selection.
    This version is updated to handle ScoredPoint objects from Qdrant.
    """
    optimized_query = generate_optimized_search_query(user_query, chat_history)
    # الآن `retrieve_from_qdrant` تعيد قائمة من كائنات ScoredPoint
    retrieved_results = retrieve_from_qdrant(optimized_query, top_k=15)

    # `generate_synthesis_prompt` يجب أن يتم تحديثها أيضاً لتتعامل مع الهيكل الجديد
    synthesis_prompt = generate_synthesis_prompt(user_query, retrieved_results, chat_history)

    recommendation_text = "I'm sorry, I had trouble processing that request. Could you try rephrasing?"
    # القائمة النهائية ستحتوي على كائنات ScoredPoint
    selected_products = []

    try:
        response = gemini_model.generate_content(synthesis_prompt)
        response_text = response.text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        # معالجة الحالة التي لا يعيد فيها الـ LLM كائن JSON صالح
        if not json_match:
            recommendation_text = response_text
            selected_products = []
        else:
            parsed_response = json.loads(json_match.group(0))
            recommendation_text = parsed_response.get("recommendation_text", "I couldn't generate a specific recommendation.")
            selected_ids_final = [str(pid) for pid in parsed_response.get("selected_product_ids", [])]

            selected_ids_set = set(selected_ids_final)
            if selected_ids_set:
                # --- التعديل الرئيسي هنا: الوصول إلى البيانات عبر .payload ---
                selected_products = [p for p in retrieved_results if str(p.payload.get('product_id')) in selected_ids_set]

    except Exception as e:
        print(f"Error during final synthesis: {e}. Falling back.")
        # في حالة الخطأ، نرجع النص الافتراضي وقائمة فارغة
        return "I'm sorry, I had trouble processing that request.", []

    # --- تحديث قسم التسجيل (Logging) ليتوافق مع الهيكل الجديد ---
    retrieved_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in retrieved_results]
    selected_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in selected_products]

    # selected_ids_final مأخوذة من استجابة الـ LLM مباشرة، لذا لا تحتاج لتعديل
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_query": user_query,
        "optimized_query": optimized_query,
        "llm_response_text": recommendation_text.replace('\n', ' '),
        "retrieved_products (id:score)": ", ".join(retrieved_ids_with_scores) if retrieved_ids_with_scores else "None",
        "selected_products (id:score)": ", ".join(selected_ids_with_scores) if selected_ids_with_scores else "None",
        "selected_product_ids_only": ", ".join(map(str, selected_ids_final)) if selected_ids_final else "None",
        "full_synthesis_prompt": synthesis_prompt.replace('\n', ' '),
    }
    log_interaction_to_csv(log_data)

    return recommendation_text, selected_products

# ==================== نهاية النسخة المعدلة ====================


# ==================== النسخة الهجينة والمحسّنة من الدالة ====================
def get_similar_products(product_id: str, similarity_threshold=0.85, min_results=3):
    try:
        point_id = create_uuid_from_string(product_id)
        # استخدام retrieve للحصول على النقطة عبر الـ ID
        source_points = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=["text"] # طلب متجه النص فقط
        )

        if not source_points:
            st.warning(f"Could not find the source product with ID: {product_id}")
            return []

        source_vectors = source_points[0].vector
        source_text_vector = source_vectors.get("text")

        if source_text_vector is None:
            st.warning(f"Could not find the text vector for product ID: {product_id}")
            return []

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name="text",
                vector=source_text_vector
            ),
            limit=20,
            with_payload=True
        )

        all_candidates = [res for res in search_results if str(res.payload.get('product_id')) != product_id]
        quality_results = [p for p in all_candidates if p.score >= similarity_threshold]

        return quality_results if quality_results else all_candidates[:min_results]

    except Exception as e:
        st.error(f"Could not find similar products: {e}")
        return []


def retrieve_by_image_vector(vector, top_k=9):
    if vector is None: return []
    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedVector(
                name="image", # --- البحث باستخدام متجه الصورة ---
                vector=vector
            ),
            limit=top_k,
            with_payload=True
        )
        return [models.ScoredPoint.model_validate(res) for res in search_results]
    except Exception as e:
        st.error(f"Error during image retrieval: {e}")
        return []



# --- Helper Function for UI ---
def display_products_in_grid(products, message_key_prefix, num_columns=3):
    if not products: return
    for i in range(0, len(products), num_columns):
        row_products = products[i:i + num_columns]
        cols = st.columns(len(row_products))
        for j, product in enumerate(row_products):
            with cols[j]:
                # --- تعديل هنا للتعامل مع كائن ScoredPoint ---
                payload = product.payload
                product_id_str = str(payload.get('product_id', ''))
                st.image(payload.get('main_image'), caption=payload.get('product_name', 'Product')[:40] + "...")
                with st.expander("View Details"):
                    st.write(f"**{payload.get('product_name')}**")
                    st.write(payload.get('combined_info'))
                # --- نهاية التعديل ---

                button_key = f"more_like_{message_key_prefix}_{product_id_str}_{i}_{j}"
                if st.button("More like this 🔁", key=button_key):
                    st.session_state.find_similar_to = product_id_str
                    st.rerun()

# ==================== الكود الكامل والمعدل لقسم الواجهة ====================

# --- 5. Streamlit User Interface ---
st.title("🛍️ Your Smart Shopping Assistant")
st.markdown("""
<span style='font-size:17px; color:#333;'>
Hello! 👋
Discover smarter shopping — tailored to your preferences, budget, and style.
Let’s make finding the perfect product effortless and inspiring. 💫
</span>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "find_similar_to" in st.session_state and st.session_state.find_similar_to:
    product_id = st.session_state.find_similar_to
    st.session_state.find_similar_to = None

    with st.spinner(f"Finding items similar to product {product_id}..."):
        similar_products = get_similar_products(product_id)

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
            recommendation_text, selected_products = get_recommendation(prompt, st.session_state.messages)

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

