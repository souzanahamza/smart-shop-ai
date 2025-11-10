# -*- coding: utf-8 -*-
"""Functions for query rewriting and optimization."""
from utils.helpers import format_chat_history


def generate_query_rewriter_prompt(user_query, chat_history):
    """Generates the prompt for query rewriting."""
    history_str = format_chat_history(chat_history)
    return f"""You are an expert search query optimization assistant. Your task is to rewrite a user's latest message into an optimal, concise search query for a vector database. Analyze the conversation history and the user's latest input to understand their true intent.
### CONVERSATION HISTORY
{history_str}
### LATEST USER INPUT
"{user_query}"
### TASK
Based on the full conversation, rewrite the latest user input into a self-contained search query. The query should be short, specific, and contain key terms. If the user's input is not a shopping query (e.g., "hello"), just return their original input.
- **CRITICAL INSTRUCTION: The final rewritten query MUST ALWAYS be in ENGLISH.** If the user's input is in another language, translate the core meaning into an English search query.
**Example 1:**
- History: "User: I need a gift for my father."
- Latest Input: "He likes waterproof watches."
- Rewritten Query: "waterproof watch for men"
**Example 2:**
- History: "User: I want a summer dress."
- Latest Input: "how about in red?"
- Rewritten Query: "red summer dress"
**Output only the rewritten query text, with no extra explanation or quotation marks.**
Rewritten Query:"""


def generate_optimized_search_query(user_query, chat_history, gemini_model):
    """Generates an optimized search query from user input using Gemini."""
    prompt = generate_query_rewriter_prompt(user_query, chat_history)
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip().replace('"', '').replace("'", "")
    except Exception as e:
        print(f"Error during query optimization: {e}. Falling back to original query.")
        return user_query

def generate_refine_translation_prompt(user_query):
    """Prompt for translation + intelligent refinement and filtering of noisy user input."""
    return f"""You are a multilingual shopping query refinement assistant.
Your job is to **translate, clean, and lightly refine** the user's input into clear, concise English suitable for product retrieval.

Core goals:
1. Preserve the user's real intent **only if it is about shopping or describing an item**.
2. Ignore or simplify irrelevant chatter (like greetings, jokes, reactions, or filler words).
3. If the message is not a shopping-related query, return a short neutral fallback like "general shopping query".
4. If the text is already in English and meaningful, keep it as-is (fix minor errors only).
5. If it's in Arabic or mixed, translate it faithfully to English.
6. Always return a short, search-friendly phrase — not a sentence.

Examples:
- "بدي فستان مثل هاد بس أزرق" → "blue dress similar to this"
- "هي شو رأيك فيها؟" → "similar item"  
- "هههه بدي شي هيك ناعم" → "elegant outfit"
- "send me link" → "general shopping query"
- "نفسه بس رجالي" → "same style but for men"
- "شي بنفس الروح" → "something with a similar vibe"
- "هاد الشي متوفر بلون تاني؟" → "available in another color?"
- "forget it 😂" → "general shopping query"

User Input:
"{user_query}"

Cleaned and Refined English Query:"""


def translate_and_refine_query_for_hybrid(user_query, gemini_model):
    """Translates, filters, and refines user text for hybrid CLIP or text-only retrieval."""
    if not user_query or not user_query.strip():
        return "general shopping query"
    prompt = generate_refine_translation_prompt(user_query)
    try:
        response = gemini_model.generate_content(prompt)
        refined = response.text.strip().replace('"', '').replace("'", "")
        # fallback if the model gives empty or weird content
        if len(refined) < 2 or refined.lower() in ["", "none", "n/a"]:
            refined = "general shopping query"
        return refined
    except Exception as e:
        print(f"[Warning] Query refinement failed: {e}")
        return "general shopping query"
