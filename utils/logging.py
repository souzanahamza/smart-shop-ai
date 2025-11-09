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
            "llm_response_text": log_data.get("llm_response_text"),
            "retrieved_products": log_data.get("retrieved_products (id:score)"),
            "selected_products": log_data.get("selected_products (id:score)"),
            "selected_product_ids_only": log_data.get("selected_product_ids_only"),
            "full_synthesis_prompt": log_data.get("full_synthesis_prompt"),
        }
        
        supabase.table("interaction_logs").insert(insert_data).execute()
    except Exception as e:
        # Silently fail to avoid disrupting the user experience
        # In production, you might want to log this to a monitoring service
        print(f"Error logging interaction to Supabase: {e}")

