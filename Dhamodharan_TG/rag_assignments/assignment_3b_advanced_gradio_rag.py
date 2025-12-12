# Assignment 3b: Advanced Gradio RAG Frontend
## Day 6 Session 2 - Building Configurable RAG Applications

# In this assignment, you'll extend your basic RAG interface with advanced configuration options to create a professional, feature-rich RAG application.

# **New Features to Add:**
# - Model selection dropdown (gpt-4o, gpt-4o-mini)
# - Temperature slider (0 to 1 with 0.1 intervals)
# - Chunk size configuration
# - Chunk overlap configuration  
# - Similarity top-k slider
# - Node postprocessor multiselect
# - Similarity cutoff slider
# - Response synthesizer multiselect

# **Learning Objectives:**
# - Advanced Gradio components and interactions
# - Dynamic RAG configuration
# - Professional UI design patterns
# - Parameter validation and handling
# - Building production-ready AI applications

# **Prerequisites:**
# - Completed Assignment 3a (Basic Gradio RAG)
# - Understanding of RAG parameters and their effects

# ## 📚 Part 1: Setup and Imports

# Import all necessary libraries including advanced RAG components for configuration options.

# **Note:** This assignment uses OpenRouter for LLM access (not OpenAI). Make sure you have your `OPENROUTER_API_KEY` environment variable set.

# Import all required libraries
import gradio as gr
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# LlamaIndex core components
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

# Advanced RAG components
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import TreeSummarize, Refine, CompactAndRefine
from llama_index.core.retrievers import VectorIndexRetriever

print("✅ All libraries imported successfully!")

## 🤖 Part 2: Advanced RAG Backend Class

# Create an advanced RAG backend that supports dynamic configuration of all parameters.

class AdvancedRAGBackend:
    """Advanced RAG backend with configurable parameters."""
    
    def __init__(self):
        self.index = None
        self.available_models = ["gpt-4o", "gpt-4o-mini"]
        self.available_postprocessors = ["SimilarityPostprocessor"]
        self.available_synthesizers = ["TreeSummarize", "Refine", "CompactAndRefine", "Default"]
        self.update_settings()
        
    def update_settings(self, model: str = "gpt-4o-mini", temperature: float = 0.1, chunk_size: int = 512, chunk_overlap: int = 50):
        """Update LlamaIndex settings based on user configuration."""
        # Set up the LLM using OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            Settings.llm = OpenRouter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        
        # Set up the embedding model (keep this constant)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            trust_remote_code=True
        )
        
        # Set chunking parameters from function parameters
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
    
    def initialize_database(self, data_folder="../data"):
        """Initialize the vector database with documents."""
        # Check if data folder exists
        if not Path(data_folder).exists():
            return f"❌ Data folder '{data_folder}' not found!"
        
        try:
            # Create vector store
            vector_store = LanceDBVectorStore(
                uri="./advanced_rag_vectordb",
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
    
    def get_postprocessor(self, postprocessor_name: str, similarity_cutoff: float):
        """Get the selected postprocessor."""
        if postprocessor_name == "SimilarityPostprocessor":
            return SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        elif postprocessor_name == "None":
            return None
        else:
            return None
    
    def get_synthesizer(self, synthesizer_name: str):
        """Get the selected response synthesizer."""
        if synthesizer_name == "TreeSummarize":
            return TreeSummarize()
        elif synthesizer_name == "Refine":
            return Refine()
        elif synthesizer_name == "CompactAndRefine":
            return CompactAndRefine()
        elif synthesizer_name == "Default":
            return None
        else:
            return None
    
    def advanced_query(self, question: str, model: str, temperature: float, 
                      chunk_size: int, chunk_overlap: int, similarity_top_k: int,
                      postprocessor_names: List[str], similarity_cutoff: float,
                      synthesizer_name: str) -> Dict[str, Any]:
        """Query the RAG system with advanced configuration."""
        
        # Check if index exists
        if self.index is None:
            return {"response": "❌ Please initialize the database first!", "sources": [], "config": {}}
        
        # Check if question is empty
        if not question or not question.strip():
            return {"response": "⚠️ Please enter a question first!", "sources": [], "config": {}}
        
        try:
            # Update settings with new parameters
            self.update_settings(model, temperature, chunk_size, chunk_overlap)
            
            # Get postprocessors
            postprocessors = []
            for name in postprocessor_names:
                processor = self.get_postprocessor(name, similarity_cutoff)
                if processor is not None:
                    postprocessors.append(processor)
            
            # Get synthesizer
            synthesizer = self.get_synthesizer(synthesizer_name)
            
            # Create query engine with all parameters
            query_engine_kwargs = {"similarity_top_k": similarity_top_k}
            if postprocessors:
                query_engine_kwargs["node_postprocessors"] = postprocessors
            if synthesizer is not None:
                query_engine_kwargs["response_synthesizer"] = synthesizer
            
            query_engine = self.index.as_query_engine(**query_engine_kwargs)
            
            # Query and get response
            response = query_engine.query(question)
            
            # Extract source information if available
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "text": node.text[:200] + "...",
                        "score": getattr(node, 'score', 0.0),
                        "source": getattr(node.node, 'metadata', {}).get('file_name', 'Unknown')
                    })
            
            return {
                "response": str(response),
                "sources": sources,
                "config": {
                    "model": model,
                    "temperature": temperature,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "similarity_top_k": similarity_top_k,
                    "postprocessors": postprocessor_names,
                    "similarity_cutoff": similarity_cutoff,
                    "synthesizer": synthesizer_name
                }
            }
        
        except Exception as e:
            return {"response": f"❌ Error processing query: {str(e)}", "sources": [], "config": {}}

# Initialize the backend
rag_backend = AdvancedRAGBackend()
print("🚀 Advanced RAG Backend initialized and ready!")

## 🎨 Part 3: Advanced Gradio Interface

import gradio as gr
# assuming you already have: import rag_backend

def create_advanced_rag_interface():
    """Create advanced RAG interface with full configuration options."""
    
    def initialize_db():
        """Handle database initialization."""
        return rag_backend.initialize_database()
    
    def handle_advanced_query(
        question, model, temperature, chunk_size, chunk_overlap, 
        similarity_top_k, postprocessors, similarity_cutoff, synthesizer
    ):
        """Handle advanced RAG queries with all configuration options."""
        result = rag_backend.advanced_query(
            question, model, temperature, chunk_size, chunk_overlap,
            similarity_top_k, postprocessors, similarity_cutoff, synthesizer
        )
        
        # Format configuration for display
        config_text = f"""**Current Configuration:**
- Model: {result['config'].get('model', 'N/A')}
- Temperature: {result['config'].get('temperature', 'N/A')}
- Chunk Size: {result['config'].get('chunk_size', 'N/A')}
- Chunk Overlap: {result['config'].get('chunk_overlap', 'N/A')}
- Similarity Top-K: {result['config'].get('similarity_top_k', 'N/A')}
- Postprocessors: {', '.join(result['config'].get('postprocessors', []))}
- Similarity Cutoff: {result['config'].get('similarity_cutoff', 'N/A')}
- Synthesizer: {result['config'].get('synthesizer', 'N/A')}"""
        
        return result["response"], config_text
    
    # Create the advanced interface structure
    with gr.Blocks(title="Advanced RAG Assistant") as interface:
        # Title and description
        gr.Markdown(
            """
            # 🧠 Advanced RAG Assistant

            Configure your Retrieval-Augmented Generation (RAG) pipeline and run queries with fine-grained control.
            
            1. **Initialize Database** to build or refresh the index.  
            2. Tune **model, temperature, chunking, retrieval, postprocessors, and synthesizer**.  
            3. Ask a question and inspect both the **response** and **current configuration**.
            """
        )
        
        # Database initialization section
        with gr.Row():
            init_btn = gr.Button("Initialize Database")
            status_output = gr.Textbox(
                label="Database Status",
                placeholder="Click 'Initialize Database' to set up the vector store / index.",
                interactive=False,
                lines=2
            )
        
        # Main layout
        with gr.Row():
            # Left column: configuration
            with gr.Column(scale=1):
                
                gr.Markdown("### ⚙️ RAG Configuration")
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    value="gpt-4o-mini"
                )
                
                # Temperature control
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.1
                )
                
                gr.Markdown("#### 📚 Chunking Parameters")
                
                # Chunking parameters
                chunk_size_input = gr.Number(
                    label="Chunk Size",
                    value=512,
                    precision=0
                )
                
                chunk_overlap_input = gr.Number(
                    label="Chunk Overlap",
                    value=50,
                    precision=0
                )
                
                gr.Markdown("#### 🔎 Retrieval Parameters")
                
                # Retrieval parameters
                similarity_topk_slider = gr.Slider(
                    label="Similarity Top-K",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5
                )
                
                # Postprocessor selection (multiselect)
                postprocessor_checkbox = gr.CheckboxGroup(
                    label="Node Postprocessors",
                    choices=["SimilarityPostprocessor"],
                    value=[]
                )
                
                # Similarity filtering
                similarity_cutoff_slider = gr.Slider(
                    label="Similarity Cutoff",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.3
                )
                
                # Response synthesizer
                synthesizer_dropdown = gr.Dropdown(
                    label="Response Synthesizer",
                    choices=["TreeSummarize", "Refine", "CompactAndRefine", "Default"],
                    value="Default"
                )
            
            # Right column: query & outputs
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Query Interface")
                
                # Query input
                query_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="Example: Summarize what the documents say about the refund policy.",
                    lines=3
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "Submit Query",
                    variant="primary"
                )
                
                # Response output
                response_output = gr.Textbox(
                    label="Model Response",
                    lines=12,
                    interactive=False
                )
                
                # Configuration display
                config_display = gr.Textbox(
                    label="Configuration Overview",
                    lines=8,
                    interactive=False
                )
        
        # Connect functions to components
        init_btn.click(initialize_db, outputs=[status_output])
        
        submit_btn.click(
            handle_advanced_query,
            inputs=[
                query_input, model_dropdown, temperature_slider,
                chunk_size_input, chunk_overlap_input, similarity_topk_slider,
                postprocessor_checkbox, similarity_cutoff_slider, synthesizer_dropdown
            ],
            outputs=[response_output, config_display]
        )
    
    return interface


# Create the interface
advanced_interface = create_advanced_rag_interface()
print("✅ Advanced RAG interface created successfully!")

## 🚀 Part 4: Launch Your Advanced Application

# Launch your advanced Gradio application and test all the configuration options!

print("🎉 Launching your Advanced RAG Assistant...")
print("🔗 Your application will open in a new browser tab!")
print("")
print("⚠️  Make sure your OPENROUTER_API_KEY environment variable is set!")
print("")
print("📋 Testing Instructions:")
print("1. Click 'Initialize Vector Database' button first")
print("2. Wait for success message")
print("3. Configure your RAG parameters:")
print("   - Choose model (gpt-4o, gpt-4o-mini)")
print("   - Adjust temperature (0.0 = deterministic, 1.0 = creative)")
print("   - Set chunk size and overlap")
print("   - Choose similarity top-k")
print("   - Select postprocessors and synthesizer")
print("4. Enter a question and click 'Ask Question'")
print("5. Review both the response and configuration used")
print("")
print("🧪 Experiments to try:")
print("- Compare different models with the same question")
print("- Test temperature effects (0.1 vs 0.9)")
print("- Try different chunk sizes (256 vs 1024)")
print("- Compare synthesizers (TreeSummarize vs Refine)")
print("- Adjust similarity cutoff to filter results")

# Your code here:
advanced_interface.launch()
## 💡 Understanding the Configuration Options

# ### Model Selection
# - **gpt-4o**: Latest and most capable model, best quality responses
# - **gpt-4o-mini**: Faster and cheaper while maintaining good quality

# ### Temperature (0.0 - 1.0)
# - **0.0-0.3**: Deterministic, factual responses
# - **0.4-0.7**: Balanced creativity and accuracy
# - **0.8-1.0**: More creative and varied responses

# ### Chunk Size & Overlap
# - **Chunk Size**: How much text to process at once (256-1024 typical)
# - **Chunk Overlap**: Overlap between chunks to maintain context (10-100 typical)

# ### Similarity Top-K (1-20)
# - **Lower values (3-5)**: More focused, faster responses
# - **Higher values (8-15)**: More comprehensive, detailed responses

# ### Node Postprocessors
# - **SimilarityPostprocessor**: Filters out low-relevance documents

# ### Similarity Cutoff (0.0-1.0)
# - **0.1-0.3**: More permissive, includes potentially relevant docs
# - **0.5-0.8**: More strict, only highly relevant docs

# ### Response Synthesizers
# - **TreeSummarize**: Hierarchical summarization, good for complex topics
# - **Refine**: Iterative refinement, builds detailed responses
# - **CompactAndRefine**: Efficient version of Refine
# - **Default**: Standard synthesis approach

# ## ✅ Assignment Completion Checklist

# Before submitting, ensure you have:

# - [ ] Set up your OPENROUTER_API_KEY environment variable
# - [ ] Imported all necessary libraries including advanced RAG components
# - [ ] Created AdvancedRAGBackend class with configurable parameters
# - [ ] Implemented all required methods:
#   - [ ] `update_settings()` - Updates LLM and chunking parameters
#   - [ ] `initialize_database()` - Sets up vector database
#   - [ ] `get_postprocessor()` - Returns selected postprocessor
#   - [ ] `get_synthesizer()` - Returns selected synthesizer
#   - [ ] `advanced_query()` - Handles queries with all configuration options
# - [ ] Created advanced Gradio interface with all required components:
#   - [ ] Initialize database button
#   - [ ] Model selection dropdown (gpt-4o, gpt-4o-mini)
#   - [ ] Temperature slider (0 to 1, step 0.1)
#   - [ ] Chunk size input (default 512)
#   - [ ] Chunk overlap input (default 50)
#   - [ ] Similarity top-k slider (1 to 20, default 5)
#   - [ ] Node postprocessor multiselect
#   - [ ] Similarity cutoff slider (0.0 to 1.0, step 0.1, default 0.3)
#   - [ ] Response synthesizer dropdown
#   - [ ] Query input and submit button
#   - [ ] Response output
#   - [ ] Configuration display
# - [ ] Connected all components to backend functions
# - [ ] Successfully launched the application
# - [ ] Tested different parameter combinations
# - [ ] Verified all configuration options work correctly

# ## 🎊 Congratulations!

# You've successfully built a professional, production-ready RAG application! You now have:

# - **Advanced Parameter Control**: Full control over all RAG system parameters
# - **Professional UI**: Clean, organized interface with proper layout
# - **Real-time Configuration**: Ability to experiment with different settings
# - **Production Patterns**: Understanding of how to build scalable AI applications

# ## 🚀 Next Steps & Extensions

# **Potential Enhancements:**
# 1. **Authentication**: Add user login and session management
# 2. **Document Upload**: Allow users to upload their own documents
# 3. **Chat History**: Implement conversation memory
# 4. **Performance Monitoring**: Add response time and quality metrics
# 5. **A/B Testing**: Compare different configurations side-by-side
# 6. **Export Features**: Download responses and configurations
# 7. **Advanced Visualizations**: Show document similarity scores and retrieval paths

# **Deployment Options:**
# - **Local**: Run on your machine for development
# - **Gradio Cloud**: Deploy with `interface.launch(share=True)`
# - **Hugging Face Spaces**: Deploy to Hugging Face for public access
# - **Docker**: Containerize for scalable deployment
# - **Cloud Platforms**: Deploy to AWS, GCP, or Azure

# You're now ready to build sophisticated AI-powered applications!
