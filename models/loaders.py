# -*- coding: utf-8 -*-
"""Functions for loading and caching ML models and API clients."""
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai
from transformers import CLIPProcessor, CLIPModel
from config import QDRANT_URL, EMBEDDING_MODEL_NAME, CLIP_MODEL_NAME, GEMINI_MODEL_NAME


@st.cache_resource
def load_models_and_clients(google_api_key, qdrant_api_key):
    """Loads and caches all the necessary models and API clients."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text Embedding Model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # Image Embedding Model (CLIP)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Qdrant Client
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=qdrant_api_key
    )

    # Generative Model
    genai.configure(api_key=google_api_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    return embedding_model, qdrant_client, gemini_model, clip_model, clip_processor

