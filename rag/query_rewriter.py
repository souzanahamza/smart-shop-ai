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

