import os
import json
from datetime import datetime

CHAT_DIR = "chats"

# Create folder if it doesn't exist
if not os.path.exists(CHAT_DIR):
    os.makedirs(CHAT_DIR)


def create_new_chat():
    chat_id = f"chat_{int(datetime.now().timestamp())}"
    path = f"{CHAT_DIR}/{chat_id}.json"

    with open(path, "w") as f:
        json.dump({"messages": []}, f, indent=4)

    return chat_id


def load_chat(chat_id):
    path = f"{CHAT_DIR}/{chat_id}.json"
    with open(path, "r") as f:
        return json.load(f)


def save_chat(chat_id, data):
    path = f"{CHAT_DIR}/{chat_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_all_chats():
    """Return all chat JSON file names sorted by timestamp."""
    return sorted(os.listdir(CHAT_DIR))
