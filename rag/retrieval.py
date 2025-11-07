# -*- coding: utf-8 -*-
"""Functions for retrieving products from Qdrant vector database."""
import streamlit as st
from qdrant_client import models
from config import COLLECTION_NAME


def retrieve_from_qdrant(query, embedding_model, qdrant_client, top_k=20):
    """Retrieves products from Qdrant based on text query."""
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


def retrieve_by_image_vector(vector, qdrant_client, top_k=9):
    """Retrieves products from Qdrant based on image vector."""
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


def get_similar_products(product_id, qdrant_client, similarity_threshold=0.85, min_results=3):
    """Finds similar products to a given product ID."""
    from utils.helpers import create_uuid_from_string
    
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

