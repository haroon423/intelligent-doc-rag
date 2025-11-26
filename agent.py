import os
import json
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import requests
from document_processor import DocumentProcessor
from vector_store import VectorStore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Direct Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def simple_groq_call(prompt, model="llama-3.3-70b-versatile"):
    """Make a simple call to Groq API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content, None
        else:
            try:
                error_detail = response.json()
                error_msg = f"API Error {response.status_code}: {json.dumps(error_detail, indent=2)}"
            except:
                error_msg = f"API Error {response.status_code}: {response.text}"
            
            return None, error_msg
    except Exception as e:
        return None, f"Error: {str(e)}"

# Test the connection
test_response, test_error = simple_groq_call("What is 2+2? Answer with just the number.")

if test_error:
    GROQ_IS_WORKING = False
    GROQ_ERROR_MSG = test_error
    GROQ_MODEL = None
else:
    GROQ_IS_WORKING = True
    GROQ_ERROR_MSG = None
    GROQ_MODEL = "llama-3.3-70b-versatile"

# Define state structure
class RAGState(TypedDict):
    query: Optional[str]
    documents: Optional[List[str]]
    retrieved_docs: Optional[List[Dict]]
    context: Optional[str]
    response: Optional[str]
    error: Optional[str]
    model_used: Optional[str]

def ingest_documents(state: RAGState):
    """Ingest documents into the vector store"""
    if "documents" not in state or not state["documents"]:
        state["error"] = "No documents provided for ingestion"
        return state
    
    try:
        # Initialize document processor and vector store
        processor = DocumentProcessor()
        vector_store = VectorStore(store_type="chroma", persist_directory="./rag_vector_db")
        
        # Process documents
        chunks = processor.process_documents(state["documents"])
        
        # Add to vector store
        success = vector_store.add_documents(chunks)
        
        if success:
            state["response"] = f"Successfully ingested {len(chunks)} document chunks from {len(state['documents'])} documents"
        else:
            state["error"] = "Failed to ingest documents"
        
        return state
    except Exception as e:
        state["error"] = f"Error during document ingestion: {str(e)}"
        return state

def retrieve_documents(state: RAGState):
    """Retrieve relevant documents based on query"""
    if "query" not in state or not state["query"]:
        state["error"] = "No query provided for retrieval"
        return state
    
    try:
        # Initialize vector store
        vector_store = VectorStore(store_type="chroma", persist_directory="./rag_vector_db")
        
        # Retrieve documents
        docs_with_scores = vector_store.similarity_search_with_score(state["query"], k=5)
        
        # Format retrieved documents
        retrieved_docs = []
        context_parts = []
        
        for doc, score in docs_with_scores:
            doc_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            }
            retrieved_docs.append(doc_info)
            context_parts.append(f"Document (relevance: {score:.4f}): {doc.page_content}")
        
        state["retrieved_docs"] = retrieved_docs
        state["context"] = "\n\n".join(context_parts)
        
        return state
    except Exception as e:
        state["error"] = f"Error during document retrieval: {str(e)}"
        return state

def generate_response(state: RAGState):
    """Generate response using retrieved context"""
    # Check for errors from previous steps
    if state.get("error"):
        state["response"] = state["error"]
        return state
    
    # Check if context is available
    if not state.get("context"):
        state["error"] = "No context available for response generation"
        return state
    
    # Check if Groq is working
    if not GROQ_IS_WORKING:
        state["error"] = "Groq API connection failed"
        return state
    
    # Create RAG prompt
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided context.
    
    CONTEXT:
    {state["context"]}
    
    QUESTION:
    {state["query"]}
    
    Based on the context provided, answer the question. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided documents."
    
    Provide a comprehensive answer with citations to the sources when possible.
    """
    
    response, error = simple_groq_call(prompt, GROQ_MODEL)
    
    if error:
        state["error"] = f"Error generating response: {error}"
        return state
    
    state["response"] = response
    state["model_used"] = GROQ_MODEL
    
    return state

# Build the RAG workflow
rag_workflow = StateGraph(RAGState)

# Add nodes
rag_workflow.add_node("ingest", ingest_documents)
rag_workflow.add_node("retrieve", retrieve_documents)
rag_workflow.add_node("generate", generate_response)

# Set entry points
rag_workflow.set_entry_point("retrieve")

# Add edges
rag_workflow.add_edge("retrieve", "generate")
rag_workflow.add_edge("generate", END)

# Create separate ingestion workflow
ingest_workflow = StateGraph(RAGState)
ingest_workflow.add_node("ingest", ingest_documents)
ingest_workflow.set_entry_point("ingest")
ingest_workflow.add_edge("ingest", END)

# Compile the graphs
rag_graph = rag_workflow.compile()
ingest_graph = ingest_workflow.compile()

# Helper functions
def ingest_documents_from_files(file_paths):
    """Ingest documents from file paths"""
    initial_state = {
        "documents": file_paths,
        "query": None,
        "retrieved_docs": None,
        "context": None,
        "response": None,
        "error": None,
        "model_used": None
    }
    
    result = ingest_graph.invoke(initial_state)
    return result.get("response", result.get("error", "No response generated"))

def query_rag(query):
    """Query the RAG system"""
    initial_state = {
        "documents": None,
        "query": query,
        "retrieved_docs": None,
        "context": None,
        "response": None,
        "error": None,
        "model_used": None
    }
    
    result = rag_graph.invoke(initial_state)
    
    return {
        "response": result.get("response", result.get("error", "No response generated")),
        "retrieved_docs": result.get("retrieved_docs", []),
        "model_used": result.get("model_used"),
        "error": result.get("error")
    }

def get_vector_store_info():
    """Get information about the vector store"""
    vector_store = VectorStore(store_type="chroma", persist_directory="./rag_vector_db")
    return vector_store.get_collection_info()

def clear_vector_store():
    """Clear the vector store"""
    vector_store = VectorStore(store_type="chroma", persist_directory="./rag_vector_db")
    return vector_store.delete_collection()

# For testing
if __name__ == "__main__":
    # Test ingestion
    print("Testing document ingestion...")
    test_files = ["./test_document.txt"]  # Replace with actual file paths
    ingest_result = ingest_documents_from_files(test_files)
    print(f"Ingestion result: {ingest_result}")
    
    # Test query
    print("\nTesting RAG query...")
    query_result = query_rag("What is the main topic of the documents?")
    print(f"Query result: {query_result}")