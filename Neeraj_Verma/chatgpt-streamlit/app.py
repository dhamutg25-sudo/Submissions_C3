import streamlit as st
from chat_manager import ChatManager
from openrouter_client import OpenRouterClient
from ui_components import (
    render_sidebar,
    display_chat_messages,
    render_summarize_button,
    inject_custom_css
)


# Page configuration
st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Note: Minimal CSS injected to enhance UX without overriding Streamlit themes


def initialize_session_state():
    """Initialize all session state variables."""
    
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = st.session_state.chat_manager.load_chat_history()
    
    if "current_chat_id" not in st.session_state:
        # Create a new chat if no history exists
        if not st.session_state.chat_history:
            new_chat = st.session_state.chat_manager.create_new_chat()
            st.session_state.current_chat_id = new_chat["id"]
            st.session_state.chat_history = [new_chat]
        else:
            st.session_state.current_chat_id = st.session_state.chat_history[0]["id"]
    
    if "messages" not in st.session_state:
        current_chat = st.session_state.chat_manager.get_chat_by_id(
            st.session_state.current_chat_id
        )
        st.session_state.messages = current_chat["messages"] if current_chat else []
    
    if "openrouter_client" not in st.session_state:
        # Load API key from secrets
        try:
            api_key = st.secrets["openrouter"]["api_key"]
            model = st.secrets["openrouter"]["model"]
            st.session_state.openrouter_client = OpenRouterClient(api_key, model)
            st.session_state.model_name = model
        except Exception as e:
            st.error(f"Error loading OpenRouter credentials: {str(e)}")
            st.stop()


def handle_new_chat():
    """Create a new chat and switch to it."""
    new_chat = st.session_state.chat_manager.create_new_chat()
    st.session_state.current_chat_id = new_chat["id"]
    st.session_state.messages = []
    st.session_state.chat_history = st.session_state.chat_manager.load_chat_history()
    st.rerun()


def handle_select_chat(chat_id: str):
    """Switch to a different chat."""
    if chat_id != st.session_state.current_chat_id:
        # Save current chat messages
        st.session_state.chat_manager.update_chat_messages(
            st.session_state.current_chat_id,
            st.session_state.messages
        )
        
        # Load selected chat
        selected_chat = st.session_state.chat_manager.get_chat_by_id(chat_id)
        if selected_chat:
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = selected_chat["messages"]
            st.rerun()


def handle_delete_chat(chat_id: str):
    """Delete a chat from history."""
    if len(st.session_state.chat_history) > 1:
        st.session_state.chat_manager.delete_chat(chat_id)
        st.session_state.chat_history = st.session_state.chat_manager.load_chat_history()
        
        # If deleted chat was current, switch to first available
        if chat_id == st.session_state.current_chat_id:
            if st.session_state.chat_history:
                st.session_state.current_chat_id = st.session_state.chat_history[0]["id"]
                st.session_state.messages = st.session_state.chat_history[0]["messages"]
        
        st.rerun()
    else:
        st.warning("Cannot delete the last chat. Create a new chat first.")


def handle_clear_chat():
    """Clear all messages from current chat."""
    st.session_state.messages = []
    st.session_state.chat_manager.clear_chat_messages(st.session_state.current_chat_id)
    st.session_state.chat_history = st.session_state.chat_manager.load_chat_history()
    st.rerun()



def handle_summarize():
    """Generate and display conversation summary."""
    if st.session_state.messages:
        with st.spinner("📝 Generating summary..."):
            summary = st.session_state.openrouter_client.summarize_conversation(
                st.session_state.messages
            )
        st.info(f"**Conversation Summary:**\n\n{summary}")
    else:
        st.warning("No messages to summarize.")


def main():
    """Main application logic."""
    
    # Initialize session state
    initialize_session_state()
    
    # Inject minimal custom CSS
    inject_custom_css()
    
    # Render sidebar
    render_sidebar(
        chat_history=st.session_state.chat_history,
        current_chat_id=st.session_state.current_chat_id,
        on_new_chat=handle_new_chat,
        on_select_chat=handle_select_chat,
        on_delete_chat=handle_delete_chat,
        on_clear_chat=handle_clear_chat,
        model_name=st.session_state.model_name
    )
    
    # Main chat area
    st.title("💬 AiChat with OpenRouter")
    
    # Display chat messages
    display_chat_messages(st.session_state.messages)
    
    # Summarize button
    render_summarize_button(
        on_summarize=handle_summarize,
        has_messages=len(st.session_state.messages) > 0
    )
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Save messages to chat history
        st.session_state.chat_manager.update_chat_messages(
            st.session_state.current_chat_id,
            st.session_state.messages
        )
        
        # Update chat history to reflect new title if needed
        st.session_state.chat_history = st.session_state.chat_manager.load_chat_history()
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("🤔 Thinking..."):
                for chunk in st.session_state.openrouter_client.chat_completion(
                    st.session_state.messages
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Save updated messages
        st.session_state.chat_manager.update_chat_messages(
            st.session_state.current_chat_id,
            st.session_state.messages
        )
        
        st.rerun()


if __name__ == "__main__":
    main()
