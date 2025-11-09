# -*- coding: utf-8 -*-
"""Test and debug utilities for Supabase connection."""
import streamlit as st
from utils.supabase_client import get_supabase_client


def test_supabase_connection():
    """Tests the Supabase connection and table access."""
    results = {
        "connection": False,
        "interaction_logs_table": False,
        "chat_history_table": False,
        "can_insert": False,
        "errors": []
    }
    
    try:
        # Test connection
        supabase = get_supabase_client()
        results["connection"] = True
        
        # Test interaction_logs table
        try:
            response = supabase.table("interaction_logs").select("id").limit(1).execute()
            results["interaction_logs_table"] = True
        except Exception as e:
            results["errors"].append(f"interaction_logs table error: {str(e)}")
        
        # Test chat_history table
        try:
            response = supabase.table("chat_history").select("id").limit(1).execute()
            results["chat_history_table"] = True
        except Exception as e:
            results["errors"].append(f"chat_history table error: {str(e)}")
        
        # Test insert capability
        try:
            test_data = {
                "original_query": "TEST_QUERY",
                "optimized_query": "TEST_OPTIMIZED",
                "llm_response_text": "TEST_RESPONSE",
                "retrieved_products": "TEST",
                "selected_products": "TEST",
                "selected_product_ids_only": "TEST",
                "full_synthesis_prompt": "TEST"
            }
            result = supabase.table("interaction_logs").insert(test_data).execute()
            # Delete test record
            if result.data and len(result.data) > 0:
                supabase.table("interaction_logs").delete().eq("id", result.data[0]["id"]).execute()
            results["can_insert"] = True
        except Exception as e:
            results["errors"].append(f"Insert test error: {str(e)}")
            
    except Exception as e:
        results["errors"].append(f"Connection error: {str(e)}")
    
    return results


def display_supabase_status():
    """Displays Supabase connection status in Streamlit."""
    with st.expander("🔍 Supabase Connection Status", expanded=False):
        test_results = test_supabase_connection()
        
        st.write("**Connection Status:**")
        if test_results["connection"]:
            st.success("✅ Connected to Supabase")
        else:
            st.error("❌ Failed to connect to Supabase")
        
        st.write("**Tables Status:**")
        if test_results["interaction_logs_table"]:
            st.success("✅ `interaction_logs` table accessible")
        else:
            st.error("❌ `interaction_logs` table not accessible")
        
        if test_results["chat_history_table"]:
            st.success("✅ `chat_history` table accessible")
        else:
            st.error("❌ `chat_history` table not accessible")
        
        st.write("**Insert Capability:**")
        if test_results["can_insert"]:
            st.success("✅ Can insert data")
        else:
            st.error("❌ Cannot insert data (check RLS policies)")
        
        if test_results["errors"]:
            st.write("**Errors:**")
            for error in test_results["errors"]:
                st.error(f"• {error}")
        
        # Show recent errors if any
        if "supabase_errors" in st.session_state and st.session_state.supabase_errors:
            st.write("**Recent Errors:**")
            for error in st.session_state.supabase_errors[-5:]:  # Show last 5 errors
                st.warning(f"• {error}")

