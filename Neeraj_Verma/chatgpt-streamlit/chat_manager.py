import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class ChatManager:
    """Manages chat history persistence using individual JSON files in a history folder."""
    
    def __init__(self, history_dir: str = "history"):
        self.history_dir = Path(history_dir)
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Create the history directory if it doesn't exist."""
        self.history_dir.mkdir(exist_ok=True)
    
    def _get_chat_file_path(self, chat_id: str) -> Path:
        """Get the file path for a specific chat."""
        return self.history_dir / f"{chat_id}.json"
    
    def _load_chat_from_file(self, chat_id: str) -> Optional[Dict]:
        """Load a single chat from its JSON file."""
        file_path = self._get_chat_file_path(chat_id)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def _save_chat_to_file(self, chat: Dict):
        """Save a single chat to its JSON file."""
        chat_id = chat["id"]
        file_path = self._get_chat_file_path(chat_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat, f, indent=2, ensure_ascii=False)
    
    def load_chat_history(self) -> List[Dict]:
        """Load all chats from individual JSON files in the history directory."""
        chats = []
        
        # Get all JSON files in the history directory
        for file_path in self.history_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chat = json.load(f)
                    chats.append(chat)
            except (json.JSONDecodeError, Exception):
                # Skip corrupted files
                continue
        
        # Sort by created_at timestamp (newest first)
        chats.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return chats
    
    def create_new_chat(self) -> Dict:
        """Create a new chat with unique ID and timestamp."""
        new_chat = {
            "id": str(uuid.uuid4()),
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        
        # Save the new chat to its own file
        self._save_chat_to_file(new_chat)
        
        return new_chat
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat by removing its JSON file."""
        file_path = self._get_chat_file_path(chat_id)
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception:
            pass
        return False
    
    def update_chat_messages(self, chat_id: str, messages: List[Dict]):
        """Update messages for a specific chat."""
        chat = self._load_chat_from_file(chat_id)
        
        if chat:
            chat["messages"] = messages
            
            # Update title based on first user message if still "New Chat"
            if chat["title"] == "New Chat" and messages:
                chat["title"] = self.get_chat_title(messages)
            
            # Save updated chat
            self._save_chat_to_file(chat)
    
    def get_chat_title(self, messages: List[Dict]) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Truncate to 50 characters
                return content[:50] + "..." if len(content) > 50 else content
        return "New Chat"
    
    def get_chat_by_id(self, chat_id: str) -> Optional[Dict]:
        """Retrieve a specific chat by ID."""
        return self._load_chat_from_file(chat_id)
    
    def clear_chat_messages(self, chat_id: str):
        """Clear all messages from a chat but keep the chat."""
        chat = self._load_chat_from_file(chat_id)
        
        if chat:
            chat["messages"] = []
            chat["title"] = "New Chat"
            self._save_chat_to_file(chat)
