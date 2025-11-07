# -*- coding: utf-8 -*-
"""Functions for loading and preparing product data from CSV."""
import streamlit as st
import pandas as pd
import json
from config import FILE_PATH


@st.cache_data
def load_and_prepare_data(file_path=FILE_PATH):
    """Loads and prepares the product data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file not found at {file_path}. Please check the path in your Colab environment.")
        st.stop()
    def clean_text(text): return str(text).replace("âœ", "").strip() if pd.notna(text) else ""
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

