# -*- coding: utf-8 -*-
"""Helper utility functions."""
import uuid


def create_uuid_from_string(text: str) -> str:
    """Creates a consistent UUID from a string product_id."""
    NAMESPACE_DNS = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(NAMESPACE_DNS, str(text)))


def format_chat_history(messages):
    """Formats chat history messages into a string."""
    history_str = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Souza"
        history_str += f"{role}: {msg['content']}\n"
    return history_str.strip()

