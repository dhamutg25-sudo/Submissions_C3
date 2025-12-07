# ChatGPT Clone with Streamlit and OpenRouter

A feature-rich ChatGPT-like application built with Streamlit and powered by OpenRouter API.

## Features

- 💬 **Chat Interface**: Clean, intuitive chat interface with user/assistant role indicators
- 📚 **Chat History**: Persistent chat history stored in JSON format
- 🗑️ **Chat Management**: Create new chats, switch between conversations, and delete old chats
- 📝 **Conversation Summarization**: On-demand conversation summaries using AI
- 🌓 **Theme Toggle**: Switch between dark and light modes
- 🧹 **Clear Chat**: Clear current conversation with one click
- 💾 **Auto-save**: All conversations automatically saved

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.streamlit/secrets.toml` file with your OpenRouter API key:

```toml
[openrouter]
api_key = "your-openrouter-api-key-here"
model = "openai/gpt-oss-120b"
```

**Important**: Never commit the `secrets.toml` file to version control!

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

- **Start a Chat**: Click "➕ New Chat" in the sidebar
- **Send Messages**: Type in the input box at the bottom and press Enter
- **Switch Chats**: Click on any chat in the history to switch to it
- **Delete Chats**: Click the 🗑️ icon next to any chat
- **Summarize**: Click "📝 Summarize Conversation" to get an AI summary
- **Clear Chat**: Use "🧹 Clear Current Chat" to reset the current conversation
- **Toggle Theme**: Use the theme toggle in Settings

## Project Structure

```
chatgpt-streamlit/
├── app.py                  # Main Streamlit application
├── chat_manager.py         # Chat history management
├── openrouter_client.py    # OpenRouter API client
├── ui_components.py        # UI components and styling
├── requirements.txt        # Python dependencies
├── chat_history.json       # Chat storage (auto-generated)
└── .streamlit/
    └── secrets.toml       # API configuration (create this)
```

## Technologies

- **Streamlit**: Web application framework
- **OpenRouter**: AI model API gateway
- **OpenAI SDK**: API client library
- **JSON**: Data persistence

## Model

This app uses the `openai/gpt-oss-120b` model via OpenRouter, a 120B parameter open-source model.

## License

MIT License
