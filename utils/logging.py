# -*- coding: utf-8 -*-
"""Logging functions for interaction tracking."""
import csv
from config import LOG_FILE


def log_interaction_to_csv(log_data):
    """Appends a dictionary of log data to a CSV file."""
    file_exists = False
    try:
        with open(LOG_FILE, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

