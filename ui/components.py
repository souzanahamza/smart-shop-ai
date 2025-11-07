# -*- coding: utf-8 -*-
"""UI component functions for displaying products and chat interface."""
import streamlit as st


def display_products_in_grid(products, message_key_prefix, num_columns=3):
    """Displays products in a grid layout."""
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

