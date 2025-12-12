# Assignment 3a: Basic Gradio RAG Frontend
## Day 6 Session 2 - Building Simple RAG Applications

# In this assignment, you'll build a simple Gradio frontend for your RAG system with just the essential features:
# - Button to initialize the vector database
# - Search query input and button
# - Display of AI responses

# **Learning Objectives:**
# - Create basic Gradio interfaces
# - Connect RAG backend to frontend
# - Handle user interactions and database initialization
# - Build functional AI-powered web applications

# **Prerequisites:**
# - Completed Assignment 1 (Vector Database Basics)
# - Completed Assignment 2 (Advanced RAG)
# - Understanding of LlamaIndex fundamentals

# ## 📚 Part 1: Setup and Imports

# Import all necessary libraries for building your Gradio RAG application.

# Import required libraries
import gradio as gr
import os
from pathlib import Path

# LlamaIndex components
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

print("✅ All libraries imported successfully!")

## 🤖 Part 2: RAG Backend Class

# Create a simple RAG backend that can initialize the database and answer queries.

class SimpleRAGBackend:
    """Simple RAG backend for Gradio frontend."""
    
    def __init__(self):
        self.index = None
        self.setup_settings()
    
    def setup_settings(self):
        """Configure LlamaIndex settings."""
        # Set up the LLM using OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            Settings.llm = OpenRouter(
                api_key=api_key,
                model="gpt-4o",
                temperature=0.1
            )
        
        # Set up the embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            trust_remote_code=True
        )
        
        # Set chunking parameters
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
    
    def initialize_database(self, data_folder="../data"):
        """Initialize the vector database with documents."""
        # Check if data folder exists
        if not Path(data_folder).exists():
            return f"❌ Data folder '{data_folder}' not found!"
        
        try:
            # Create vector store
            vector_store = LanceDBVectorStore(
                uri="./basic_rag_vectordb",
                table_name="documents"
            )
            
            # Load documents
            reader = SimpleDirectoryReader(input_dir=data_folder, recursive=True)
            documents = reader.load_data()
            
            # Create storage context and index
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                show_progress=True
            )
            
            return f"✅ Database initialized successfully with {len(documents)} documents!"
        
        except Exception as e:
            return f"❌ Error initializing database: {str(e)}"

    def query(self, question):
        """Query the RAG system and return response."""
        # Check if index exists
        if self.index is None:
            return "❌ Please initialize the database first!"
        
        # Check if question is empty
        if not question or not question.strip():
            return "⚠️ Please enter a question first!"
        
        try:
            # Create query engine and get response
            query_engine = self.index.as_query_engine()
            response = query_engine.query(question)
            return str(response)
        
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

# Initialize the backend
rag_backend = SimpleRAGBackend()
print("🚀 RAG Backend initialized and ready!")

## 🎨 Part 3: Gradio Interface

import gradio as gr
# assuming you already have: import rag_backend

def create_basic_rag_interface():
    """Create basic RAG interface with essential features."""
    
    def initialize_db():
        """Handle database initialization."""
        return rag_backend.initialize_database()
    
    def handle_query(question):
        """Handle user queries."""
        return rag_backend.query(question)
    
    with gr.Blocks(title="Basic RAG Assistant") as interface:
        # Title and description
        gr.Markdown(
            """
            # 🧠 Basic RAG Assistant
            
            1. Click **Initialize Database** to set up or refresh the knowledge base.  
            2. Type your question in the box below.  
            3. Click **Submit Query** to get an answer.
            """
        )

        # Initialization section
        with gr.Row():
            init_btn = gr.Button("Initialize Database")
            status_output = gr.Textbox(
                label="Status",
                placeholder="Click 'Initialize Database' to set up the index.",
                interactive=False
            )

        # Query section
        gr.Markdown("## 🔍 Ask a Question")
        query_input = gr.Textbox(
            label="Your question",
            lines=3,
            placeholder="Example: What does the document say about refund policy?"
        )
        submit_btn = gr.Button("Submit Query")
        response_output = gr.Textbox(
            label="Response",
            lines=8
        )
        
        # Connect buttons to functions
        init_btn.click(initialize_db, outputs=[status_output])
        submit_btn.click(handle_query, inputs=[query_input], outputs=[response_output])
        
    return interface


# Create the interface
basic_interface = create_basic_rag_interface()
print("✅ Basic RAG interface created successfully!")

## 🚀 Part 4: Launch Your Application

# Launch your Gradio application and test it!

print("🎉 Launching your Basic RAG Assistant...")
print("🔗 Your application will open in a new browser tab!")
print("")
print("📋 Testing Instructions:")
print("1. Click 'Initialize Database' button first")
print("2. Wait for success message")
print("3. Enter a question in the query box")
print("4. Click 'Ask Question' to get AI response")
print("")
print("💡 Example questions to try:")
print("- What are the main topics in the documents?")
print("- Summarize the key findings")
print("- Explain the methodology used")
print("")
print("🚀 Launch your app:")

# Your launch code here:
# Uncomment when implemented
basic_interface.launch()
## ✅ Assignment Completion Checklist

# Before submitting, ensure you have:

# - [x] RAG backend is provided and working
# - [ ] Created Gradio interface with required components:
#   - [ ] Title and description using gr.Markdown()
#   - [ ] Initialize database button using gr.Button()
#   - [ ] Status output using gr.Textbox()
#   - [ ] Query input field using gr.Textbox()
#   - [ ] Submit query button using gr.Button()
#   - [ ] Response output area using gr.Textbox()
# - [ ] Connected buttons to backend functions using .click()
# - [ ] Successfully launched the application
# - [ ] Tested the full workflow (initialize → query → response)

## 🎊 Congratulations!

# You've successfully built your first Gradio RAG application! You now have:

# - A functional web interface for your RAG system
# - Understanding of Gradio basics and component connections
# - A foundation for building more complex AI applications

# **Next Steps**: Complete Assignment 3b to add advanced configuration options to your RAG interface!
