# -*- coding: utf-8 -*-
"""Logging functions for interaction tracking."""
import streamlit as st
from utils.supabase_client import get_supabase_client


def log_interaction_to_csv(log_data):
    """
    Logs interaction data to Supabase instead of CSV.
    Maintains the same function signature for backward compatibility.
    """
    try:
        supabase = get_supabase_client()
        
        # Map the log_data dictionary to Supabase table columns
        insert_data = {
            "timestamp": log_data.get("timestamp"),
            "original_query": log_data.get("original_query"),
            "optimized_query": log_data.get("optimized_query"),
            "retrieval_mode": log_data.get("retrieval_mode"),
            "llm_response_text": log_data.get("llm_response_text"),
            "retrieved_products": log_data.get("retrieved_products (id:score)"),
            "selected_products": log_data.get("selected_products (id:score)"),
            "selected_product_ids_only": log_data.get("selected_product_ids_only"),
            "full_synthesis_prompt": log_data.get("full_synthesis_prompt"),
        }
        
        result = supabase.table("interaction_logs").insert(insert_data).execute()
        if hasattr(st, 'session_state'):
            if "supabase_debug" in st.session_state:
                st.session_state.supabase_debug.append(f"✅ Logged interaction: {log_data.get('original_query', 'N/A')[:50]}")
    except Exception as e:
        # Log error for debugging
        error_msg = f"Error logging interaction to Supabase: {str(e)}"
        print(error_msg)
        if hasattr(st, 'session_state'):
            if "supabase_errors" not in st.session_state:
                st.session_state.supabase_errors = []
            st.session_state.supabase_errors.append(error_msg)

