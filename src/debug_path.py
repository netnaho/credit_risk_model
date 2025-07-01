# src/debug_path.py

import os
from pathlib import Path

print("--- Starting Path Debugger ---")

try:
    # Define the project root directory programmatically
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'transactions.csv'

    print(f"Calculated Project Root: {PROJECT_ROOT}")
    print(f"Calculated Data Path:    {DATA_PATH}")

    # Check if the file exists from Python's perspective
    file_exists = os.path.exists(DATA_PATH)
    print(f"\nDoes the file exist at this path? -> {file_exists}")

    if file_exists:
        # Check if we have read permissions
        can_read = os.access(DATA_PATH, os.R_OK)
        print(f"Can we read the file? -> {can_read}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("--- Debugger Finished ---")