# -*- coding: utf-8 -*-
"""Functions for generating recommendations and synthesizing responses."""
import json
import re
from datetime import datetime
from utils.helpers import format_chat_history
from utils.logging import log_interaction_to_csv


def generate_synthesis_prompt(user_query, retrieved_payloads, chat_history):
    """Generates the prompt for response synthesis."""
    history_str = format_chat_history(chat_history)
    context_str = "\n\n".join([f"ID: {p.payload.get('product_id', 'N/A')}\nName: {p.payload.get('product_name', 'N/A')}" for p in retrieved_payloads])
    # context_str = "\n\n".join([f"--- Product ---\nID: {p.get('product_id', 'N/A')}\nName: {p.get('product_name', 'N/A')}\nDetails: {p.get('combined_info', 'N/A')}" for p in retrieved_payloads])
    return f"""You are **Souza** SmartShop Assistant, an expert AI shopping assistant.
---
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


def get_recommendation(user_query: str, chat_history: list, embedding_model, qdrant_client, gemini_model):
    """
    Orchestrates the recommendation process: gets a query, retrieves products,
    and uses an LLM to generate a final recommendation text and product selection.
    This version is updated to handle ScoredPoint objects from Qdrant.
    """
    from rag.query_rewriter import generate_optimized_search_query
    from rag.retrieval import retrieve_from_qdrant
    
    optimized_query = generate_optimized_search_query(user_query, chat_history, gemini_model)
    # الآن `retrieve_from_qdrant` تعيد قائمة من كائنات ScoredPoint
    retrieved_results = retrieve_from_qdrant(optimized_query, embedding_model, qdrant_client, top_k=15)

    # `generate_synthesis_prompt` يجب أن يتم تحديثها أيضاً لتتعامل مع الهيكل الجديد
    synthesis_prompt = generate_synthesis_prompt(user_query, retrieved_results, chat_history)

    recommendation_text = "I'm sorry, I had trouble processing that request. Could you try rephrasing?"
    # القائمة النهائية ستحتوي على كائنات ScoredPoint
    selected_products = []

    try:
        response = gemini_model.generate_content(synthesis_prompt)
        response_text = response.text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        # معالجة الحالة التي لا يعيد فيها الـ LLM كائن JSON صالح
        if not json_match:
            recommendation_text = response_text
            selected_products = []
        else:
            parsed_response = json.loads(json_match.group(0))
            recommendation_text = parsed_response.get("recommendation_text", "I couldn't generate a specific recommendation.")
            selected_ids_final = [str(pid) for pid in parsed_response.get("selected_product_ids", [])]

            selected_ids_set = set(selected_ids_final)
            if selected_ids_set:
                # --- التعديل الرئيسي هنا: الوصول إلى البيانات عبر .payload ---
                selected_products = [p for p in retrieved_results if str(p.payload.get('product_id')) in selected_ids_set]

    except Exception as e:
        print(f"Error during final synthesis: {e}. Falling back.")
        # في حالة الخطأ، نرجع النص الافتراضي وقائمة فارغة
        return "I'm sorry, I had trouble processing that request.", []

    # --- تحديث قسم التسجيل (Logging) ليتوافق مع الهيكل الجديد ---
    retrieved_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in retrieved_results]
    selected_ids_with_scores = [f"{p.payload.get('product_id')}:{p.score:.4f}" for p in selected_products]

    # selected_ids_final مأخوذة من استجابة الـ LLM مباشرة، لذا لا تحتاج لتعديل
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_query": user_query,
        "optimized_query": optimized_query,
        "llm_response_text": recommendation_text.replace('\n', ' '),
        "retrieved_products (id:score)": ", ".join(retrieved_ids_with_scores) if retrieved_ids_with_scores else "None",
        "selected_products (id:score)": ", ".join(selected_ids_with_scores) if selected_ids_with_scores else "None",
        "selected_product_ids_only": ", ".join(map(str, selected_ids_final)) if selected_ids_final else "None",
        "full_synthesis_prompt": synthesis_prompt.replace('\n', ' '),
    }
    log_interaction_to_csv(log_data)

    return recommendation_text, selected_products

