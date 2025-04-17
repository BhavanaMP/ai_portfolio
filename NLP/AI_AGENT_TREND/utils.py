# utils.py (Helper Functions)
import os


def save_to_file(filename, content):
    """Saves text content to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def load_from_file(filename):
    """Loads text content from a file."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    return None
