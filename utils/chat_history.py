# -*- coding: utf-8 -*-
"""Chat history persistence utilities using Supabase."""
import streamlit as st
import json
from utils.supabase_client import get_supabase_client


def serialize_products(products):
    """
    Converts ScoredPoint objects (or any product objects) to serializable format.
    Extracts payload and score for storage.
    """
    if not products:
        return []
    
    serialized = []
    for product in products:
        # Handle ScoredPoint objects from Qdrant
        if hasattr(product, 'payload') and hasattr(product, 'score'):
            serialized.append({
                "payload": product.payload,
                "score": float(product.score) if product.score is not None else None
            })
        # Handle already serialized dicts
        elif isinstance(product, dict):
            serialized.append(product)
        else:
            # Fallback: try to convert to dict
            serialized.append({"payload": dict(product) if hasattr(product, '__dict__') else product})
    return serialized


def deserialize_products(products_data):
    """
    Converts serialized product data back to a format compatible with the UI.
    Returns a list of objects with .payload attribute (mimicking ScoredPoint).
    """
    if not products_data:
        return []
    
    # Create a simple class to mimic ScoredPoint behavior
    class ProductWrapper:
        def __init__(self, payload, score=None):
            self.payload = payload
            self.score = score
    
    deserialized = []
    for item in products_data:
        if isinstance(item, dict):
            payload = item.get("payload", item)
            score = item.get("score")
            deserialized.append(ProductWrapper(payload, score))
        else:
            # Fallback
            deserialized.append(ProductWrapper(item))
    
    return deserialized


def get_session_id() -> str:
    """
    Generates or retrieves a stable session ID for the current user session.
    Uses Streamlit's session state to maintain consistency.
    """
    if "chat_session_id" not in st.session_state:
        # Generate a unique session ID (you could also use user info if available)
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())
    return st.session_state.chat_session_id


def load_chat_history_from_supabase() -> list:
    """
    Loads chat history from Supabase for the current session.
    Returns a list of messages in the same format as st.session_state.messages.
    """
    try:
        supabase = get_supabase_client()
        session_id = get_session_id()
        
        # Fetch messages ordered by message_index
        response = supabase.table("chat_history")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("message_index", desc=False)\
            .execute()
        
        messages = []
        for row in response.data:
            # Handle products - Supabase returns JSONB as Python objects, but handle both cases
            products_data = row.get("products", [])
            if isinstance(products_data, str):
                products_data = json.loads(products_data) if products_data else []
            elif products_data is None:
                products_data = []
            
            # Deserialize products back to usable format
            products = deserialize_products(products_data)
            
            message = {
                "role": row["role"],
                "content": row["content"],
                "products": products
            }
            messages.append(message)
        
        return messages
    except Exception as e:
        # If there's an error, return empty list (graceful degradation)
        error_msg = f"Error loading chat history from Supabase: {str(e)}"
        print(error_msg)
        if hasattr(st, 'session_state'):
            if "supabase_errors" not in st.session_state:
                st.session_state.supabase_errors = []
            st.session_state.supabase_errors.append(error_msg)
        return []


def save_message_to_supabase(message: dict, message_index: int):
    """
    Saves a single message to Supabase.
    
    Args:
        message: Dictionary with 'role', 'content', and optionally 'products'
        message_index: The index of this message in the conversation
    """
    try:
        supabase = get_supabase_client()
        session_id = get_session_id()
        
        # Serialize products to make them JSON-compatible
        products = message.get("products", [])
        serialized_products = serialize_products(products)
        
        insert_data = {
            "session_id": session_id,
            "message_index": message_index,
            "role": message["role"],
            "content": message["content"],
            "products": serialized_products
        }
        
        # Use upsert to handle updates if message_index already exists
        result = supabase.table("chat_history")\
            .upsert(insert_data, on_conflict="session_id,message_index")\
            .execute()
        
        # Debug logging
        if hasattr(st, 'session_state'):
            if "supabase_debug" not in st.session_state:
                st.session_state.supabase_debug = []
            st.session_state.supabase_debug.append(f"✅ Saved message {message_index}: {message.get('content', '')[:50]}")
    except Exception as e:
        # Log error for debugging
        error_msg = f"Error saving message to Supabase: {str(e)}"
        print(error_msg)
        if hasattr(st, 'session_state'):
            if "supabase_errors" not in st.session_state:
                st.session_state.supabase_errors = []
            st.session_state.supabase_errors.append(error_msg)


def save_chat_history_to_supabase(messages: list):
    """
    Saves the entire chat history to Supabase.
    This is called after each message is added to ensure persistence.
    """
    for index, message in enumerate(messages):
        save_message_to_supabase(message, index)

