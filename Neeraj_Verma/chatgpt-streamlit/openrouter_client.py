import streamlit as st
from openai import OpenAI
from typing import List, Dict, Generator


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def chat_completion(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        Send messages to OpenRouter and stream the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Yields:
            Chunks of the assistant's response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://streamlit-chatbot-app.local",
                    "X-Title": "Streamlit ChatGPT Clone"
                }
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def chat_completion_non_streaming(self, messages: List[Dict]) -> str:
        """
        Send messages to OpenRouter and get complete response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Complete assistant response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_headers={
                    "HTTP-Referer": "https://streamlit-chatbot-app.local",
                    "X-Title": "Streamlit ChatGPT Clone"
                }
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def summarize_conversation(self, messages: List[Dict]) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            messages: List of message dictionaries from the conversation
            
        Returns:
            Summary text
        """
        if not messages:
            return "No messages to summarize."
        
        # Create summarization prompt
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        summary_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise summaries of conversations. Provide a brief summary highlighting the main topics discussed and key points."
            },
            {
                "role": "user",
                "content": f"Please summarize this conversation:\n\n{conversation_text}"
            }
        ]
        
        return self.chat_completion_non_streaming(summary_messages)
