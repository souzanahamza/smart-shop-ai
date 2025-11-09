# -*- coding: utf-8 -*-
"""Supabase client initialization and utilities."""
import streamlit as st
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Initializes and returns a Supabase client using secrets."""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"FATAL ERROR: Supabase configuration not found. Missing: {e}")
        st.stop()

