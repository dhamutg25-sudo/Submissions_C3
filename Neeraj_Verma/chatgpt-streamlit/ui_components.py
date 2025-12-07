import streamlit as st
from typing import List, Dict, Callable
from datetime import datetime


def render_sidebar(
    chat_history: List[Dict],
    current_chat_id: str,
    on_new_chat: Callable,
    on_select_chat: Callable,
    on_delete_chat: Callable,
    on_clear_chat: Callable,
    model_name: str
):
    """Render the complete sidebar with all controls."""
    
    with st.sidebar:
        st.title("💬 Conversations")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            on_new_chat()
        
        st.divider()
        
        # Chat History
        st.subheader("Chat History")
        
        if chat_history:
            for chat in chat_history:
                # Use columns with vertical centering
                cols = st.columns([0.85, 0.15])
                
                with cols[0]:
                    is_current = chat["id"] == current_chat_id
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        f"{'📌 ' if is_current else '💭 '}{chat['title'][:30]}",
                        key=f"chat_{chat['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        on_select_chat(chat["id"])
                
                with cols[1]:
                    if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat", use_container_width=True):
                        on_delete_chat(chat["id"])
        else:
            st.info("No chat history yet. Start a new chat!")
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.write(f"**Model:** {model_name}")
            st.caption("💡 To change theme: Click ☰ menu → Settings → Theme")
        
        st.divider()
        
        # Clear Current Chat
        if st.button("🧹 Clear Current Chat", use_container_width=True):
            on_clear_chat()


def render_message(role: str, content: str):
    """Render a single chat message using Streamlit's native components."""
    
    # Use Streamlit's built-in chat message component which handles theming automatically
    with st.chat_message(role):
        st.markdown(content)


def render_summarize_button(on_summarize: Callable, has_messages: bool):
    """Render the conversation summary button."""
    
    if has_messages:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📝 Summarize Conversation", use_container_width=True):
                on_summarize()
    else:
        st.info("Start a conversation to see the summary option.")


def display_chat_messages(messages: List[Dict]):
    """Display all messages in the chat."""
    
    for message in messages:
        render_message(message["role"], message["content"])


def inject_custom_css():
    """Inject minimal custom CSS that works with Streamlit's default theming."""
    
    st.markdown(
        """
        <style>
        /* Button hover effects */
        .stButton>button {
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Chat input styling */
        .stChatInput {
            border-radius: 10px;
        }
        
        /* Smooth transitions */
        .element-container {
            transition: opacity 0.2s ease;
        }
        
        /* Vertically center sidebar buttons */
        [data-testid="stSidebar"] [data-testid="column"] {
            display: flex !important;
            align-items: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
