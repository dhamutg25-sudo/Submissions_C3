import streamlit as st
from openai import OpenAI
import os
import json
import glob
from datetime import datetime
from dotenv import load_dotenv

# 1. Configuration and Setup
load_dotenv() # Load API key from .env file

# Page Config (Tab title and icon)
st.set_page_config(page_title="OpenRouter Chat", page_icon="🤖", layout="wide")

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "openai/gpt-oss-120b"
CHATS_DIR = "chats"

# Ensure chats directory exists
if not os.path.exists(CHATS_DIR):
    os.makedirs(CHATS_DIR)

# --- Helper Functions ---

def load_chat(filename):
    """Loads a specific JSON chat file into session state."""
    filepath = os.path.join(CHATS_DIR, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
        st.session_state.messages = data["messages"]
        st.session_state.current_chat_file = filename

def save_chat():
    """Saves the current session state messages to the JSON file."""
    if "current_chat_file" not in st.session_state:
        # Generate a new filename based on timestamp if one doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.current_chat_file = f"chat_{timestamp}.json"
    
    filepath = os.path.join(CHATS_DIR, st.session_state.current_chat_file)
    
    # Structure to save
    chat_data = {
        "title": st.session_state.messages[0]["content"][:30] if st.session_state.messages else "New Chat",
        "messages": st.session_state.messages
    }
    
    with open(filepath, "w") as f:
        json.dump(chat_data, f, indent=4)

def create_new_chat():
    """Resets the state for a new conversation."""
    st.session_state.messages = []
    if "current_chat_file" in st.session_state:
        del st.session_state.current_chat_file

def delete_chat(filename):
    """Deletes a JSON file."""
    os.remove(os.path.join(CHATS_DIR, filename))
    # If the deleted chat was open, reset
    if "current_chat_file" in st.session_state and st.session_state.current_chat_file == filename:
        create_new_chat()
    st.rerun()

def get_summary():
    """Generates a summary of the current conversation."""
    if not st.session_state.messages:
        return "No conversation to summarize."
    
    # Create a temporary prompt for summary
    summary_messages = st.session_state.messages.copy()
    summary_messages.append({"role": "user", "content": "Summarize our conversation so far in 3 bullet points."})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=summary_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

# --- UI Layout ---

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Sidebar (History & New Chat)
with st.sidebar:
    st.title("💬 Conversations")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
    
    st.markdown("---")
    st.subheader("Chat History")
    
    # List JSON files in the directory
    files = sorted(glob.glob(os.path.join(CHATS_DIR, "*.json")), reverse=True)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Load the title from the file usually, or just use filename
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                display_name = data.get("title", filename)
        except:
            display_name = filename

        # Layout for history item: Name + Delete button
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(display_name, key=f"load_{filename}", use_container_width=True):
                load_chat(filename)
        with col2:
            if st.button("🗑️", key=f"del_{filename}"):
                delete_chat(filename)

# 2. Main Chat Area
st.title("☺ Hello")

# Summary Expandable Section
with st.expander("📝 Summarize Conversation"):
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary_text = get_summary()
            st.markdown(summary_text)

# Display Chat Messages
for msg in st.session_state.messages:
    # We map specific avatars if needed, or use default
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # If there are reasoning details stored (from your snippet), we can show them
        if "reasoning_details" in msg and msg["reasoning_details"]:
            with st.status("Reasoning Process", state="complete"):
                st.write(msg["reasoning_details"])

# 3. Chat Input & Processing
if prompt := st.chat_input("What would you like to know?"):
    # 3a. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # 3b. API Call
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        try:
            # 1. API Call
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages
                ],
                extra_body={"reasoning": {"enabled": True}}
            )
            
            # 2. Check for a valid response choice
            if response.choices and response.choices[0].message:
                
                # Safely get the main content
                full_response = response.choices[0].message.content
                
                # Safely get reasoning details (will be None if not present)
                reasoning_data = getattr(response.choices[0].message, 'reasoning_details', None)

                message_placeholder.markdown(full_response)
                
                # 3. Append assistant message to history
                assistant_msg = {
                    "role": "assistant", 
                    "content": full_response or "An empty response was received.", # Use empty string if content is None
                    "reasoning_details": reasoning_data 
                }
                st.session_state.messages.append(assistant_msg)
                
                # 4. Save to JSON
                save_chat()

            else:
                st.error("Error: The API returned an empty or invalid response structure.")
                
        except Exception as e:
            st.error(f"API Request Failed: {e}")