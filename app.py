import streamlit as st
import os
import tempfile
from agent import (
    ingest_documents_from_files, 
    query_rag, 
    get_vector_store_info, 
    clear_vector_store,
    GROQ_IS_WORKING,
    GROQ_MODEL
)
from document_processor import DocumentProcessor

st.set_page_config(
    page_title="LangGraph-Groq RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 LangGraph-Groq RAG System")
st.markdown("Upload documents and ask questions about them using AI-powered retrieval")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Show Groq status
if GROQ_IS_WORKING:
    st.sidebar.success(f"✅ Groq API is working! Using model: {GROQ_MODEL}")
else:
    st.sidebar.error("❌ Groq API connection failed!")

# Vector store info
vector_store_info = get_vector_store_info()
st.sidebar.subheader("📊 Vector Store Info")
st.sidebar.json(vector_store_info)

# Clear vector store button
if st.sidebar.button("🗑️ Clear Vector Store"):
    if clear_vector_store():
        st.sidebar.success("Vector store cleared successfully!")
        st.rerun()
    else:
        st.sidebar.error("Failed to clear vector store")

# Main content
tab1, tab2 = st.tabs(["📄 Document Management", "❓ Q&A"])

with tab1:
    st.header("📄 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    # Process uploaded files
    if uploaded_files and st.button("📥 Process Documents"):
        if not GROQ_IS_WORKING:
            st.error("Cannot process documents - Groq API is not connected.")
        else:
            with st.spinner("Processing documents..."):
                # Save uploaded files to temporary directory
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Ingest documents
                result = ingest_documents_from_files(file_paths)
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir)
                
                if "error" in result.lower():
                    st.error(f"Error: {result}")
                else:
                    st.success(result)
                    st.rerun()
    
    # Manual text input
    st.subheader("✏️ Add Text Manually")
    manual_text = st.text_area("Enter text to add to the knowledge base:", height=200)
    
    if manual_text and st.button("➕ Add Text"):
        if not GROQ_IS_WORKING:
            st.error("Cannot add text - Groq API is not connected.")
        else:
            with st.spinner("Processing text..."):
                # Create a temporary file with the text
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, "manual_input.txt")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(manual_text)
                
                # Ingest document
                result = ingest_documents_from_files([file_path])
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir)
                
                if "error" in result.lower():
                    st.error(f"Error: {result}")
                else:
                    st.success(result)
                    st.rerun()

with tab2:
    st.header("❓ Q&A")
    
    # Query input
    query = st.text_input("Ask a question about your documents:", key="query_input")
    
    if query and st.button("🔍 Get Answer"):
        if not GROQ_IS_WORKING:
            st.error("Cannot answer questions - Groq API is not connected.")
        else:
            with st.spinner("Searching documents and generating answer..."):
                result = query_rag(query)
                
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result["response"])
                    
                    # Display model used
                    st.caption(f"Generated using: {result.get('model_used', 'Unknown model')}")
                    
                    # Display retrieved documents
                    if result.get("retrieved_docs"):
                        with st.expander("📚 Retrieved Documents"):
                            for i, doc in enumerate(result["retrieved_docs"]):
                                st.markdown(f"**Document {i+1}** (Relevance: {doc['relevance_score']:.4f})")
                                st.write(doc["content"])
                                st.caption(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                                st.divider()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using LangGraph + Groq")