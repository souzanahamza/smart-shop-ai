# -*- coding: utf-8 -*-
"""Functions for generating recommendations and synthesizing responses."""
import json
import re
from datetime import datetime
from utils.helpers import format_chat_history
from utils.logging import log_interaction_to_csv


def generate_synthesis_prompt(user_query, retrieved_payloads, chat_history , optimized_query):
    """Generates the prompt for response synthesis."""
    history_str = format_chat_history(chat_history)
    context_str = "\n\n".join([f"ID: {p.payload.get('product_id', 'N/A')}\nName: {p.payload.get('product_name', 'N/A')}" for p in retrieved_payloads])
    # 🔸 إضافة ملاحظة توضح حالة الإدخال
    visual_context_note = ""
    if optimized_query:
        if "[HYBRID_CLIP]" in optimized_query:
            visual_context_note = (
                "The user uploaded an image **and also provided text** describing what they want. "
                "The retrieved products reflect both the visual and textual meaning. "
                "When generating your recommendation, assume you can interpret the image content and its described intent together."
            )
        elif "[IMAGE_SEARCH]" in optimized_query:
            visual_context_note = (
                "The user uploaded an image without additional text. "
                "The retrieved products are visually similar to the uploaded image. "
                "Respond as if you can see what the image contains."
            )

    # context_str = "\n\n".join([f"--- Product ---\nID: {p.get('product_id', 'N/A')}\nName: {p.get('product_name', 'N/A')}\nDetails: {p.get('combined_info', 'N/A')}" for p in retrieved_payloads])
    return f"""You are **Souza** SmartShop Assistant, an expert AI shopping assistant.
---
{visual_context_note}
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


def get_recommendation(
    user_query: str,
    chat_history: list,
    embedding_model,
    qdrant_client,
    gemini_model,
    image_bytes=None,
    clip_model=None,
    clip_processor=None
):
    """
    Orchestrates the recommendation process using text, image, or hybrid (CLIP) input.
    Keeps the original text logic fully intact and extends to handle image-based retrieval.
    """
    import torch
    import numpy as np
    from PIL import Image
    import json
    import re
    from datetime import datetime
    from rag.query_rewriter import generate_optimized_search_query ,translate_and_refine_query_for_hybrid
    from rag.retrieval import retrieve_from_qdrant , retrieve_by_image_vector , retrieve_hybrid_clip
    from utils.logging import log_interaction_to_csv
    from utils.helpers import format_chat_history
    from rag.synthesis import generate_synthesis_prompt

    # --- 1. تحويل الصورة إلى متجه (اختياري) ---
    image_vector = None
    if image_bytes and clip_model and clip_processor:
        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            clip_inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_features = clip_model.get_image_features(**clip_inputs).squeeze().cpu().numpy()
                image_vector = img_features / np.linalg.norm(img_features)
        except Exception as e:
            print(f"[Warning] Could not process image: {e}")

    # --- 2. اختيار نوع البحث ---
    if image_vector is None:  # 🔹 نص فقط (السلوك القديم)
        optimized_query = generate_optimized_search_query(user_query, chat_history, gemini_model)
        retrieved_results = retrieve_from_qdrant(optimized_query, embedding_model, qdrant_client, top_k=15)
        retrieval_mode = "TEXT_ONLY"


    elif user_query and user_query.strip():  # 🔹 نص + صورة
        # 🔸 1. ترجمة وتحسين النص (Refine + Translate)
        refined_query = translate_and_refine_query_for_hybrid(user_query, gemini_model)

        # 🔸 2. دمج النص المترجم مع متجه الصورة ضمن CLIP space
        retrieved_results = retrieve_hybrid_clip(
            refined_query,  # استخدم النسخة المحسّنة من النص
            image_vector,
            clip_model,
            clip_processor,
            qdrant_client,
            alpha=0.4,
            top_k=15
        )

        optimized_query = f"[HYBRID_CLIP] {refined_query}"
        retrieval_mode = "HYBRID"



    else:  
        retrieved_results = retrieve_by_image_vector(image_vector, qdrant_client, top_k=15)
        optimized_query = "[IMAGE_SEARCH]"
        retrieval_mode = "IMAGE_ONLY"

    synthesis_prompt = generate_synthesis_prompt(user_query, retrieved_results, chat_history , optimized_query)

    recommendation_text = "I'm sorry, I had trouble processing that request. Could you try rephrasing?"
    selected_products = []

    try:
        response = gemini_model.generate_content(synthesis_prompt)
        response_text = response.text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if not json_match:
            recommendation_text = response_text
            selected_products = []
        else:
            parsed_response = json.loads(json_match.group(0))
            recommendation_text = parsed_response.get(
                "recommendation_text", "I couldn't generate a specific recommendation."
            )
            selected_ids_final = [str(pid) for pid in parsed_response.get("selected_product_ids", [])]

            selected_ids_set = set(selected_ids_final)
            if selected_ids_set:
                selected_products = [
                    p for p in retrieved_results if str(p.payload.get("product_id")) in selected_ids_set
                ]

    except Exception as e:
        print(f"Error during final synthesis: {e}. Falling back.")
        return "I'm sorry, I had trouble processing that request.", []

    retrieved_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in retrieved_results]
    selected_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in selected_products]

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_query": user_query,
        "optimized_query": optimized_query,
        "retrieval_mode": retrieval_mode,
        "llm_response_text": recommendation_text.replace('\n', ' '),
        "retrieved_products (id:score)": ", ".join(retrieved_ids_with_scores) if retrieved_ids_with_scores else "None",
        "selected_products (id:score)": ", ".join(selected_ids_with_scores) if selected_ids_with_scores else "None",
        "selected_product_ids_only": ", ".join(map(str, selected_ids_final)) if selected_ids_final else "None",
        "full_synthesis_prompt": synthesis_prompt.replace('\n', ' '),
    }

    log_interaction_to_csv(log_data)

    return recommendation_text, selected_products
