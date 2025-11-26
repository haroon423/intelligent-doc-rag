import os
from typing import List, Dict, Any, Optional, Tuple
# CORRECT IMPORTS:
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # FIXED: Chroma is in community
from langchain_community.vectorstores import FAISS
import shutil

class VectorStore:
    """Handle vector database operations for RAG"""
    
    def __init__(self, store_type="chroma", persist_directory="./vector_db"):
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize vector database"""
        if self.store_type == "chroma":
            if os.path.exists(self.persist_directory):
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                self.db = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
        elif self.store_type == "faiss":
            index_path = os.path.join(self.persist_directory, "faiss_index")
            if os.path.exists(index_path):
                self.db = FAISS.load_local(
                    index_path, 
                    self.embeddings
                )
            else:
                # Create empty FAISS index
                self.db = None
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            if self.store_type == "chroma":
                if self.db is None:
                    self.db = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                else:
                    self.db.add_documents(documents)
                self.db.persist()
            elif self.store_type == "faiss":
                if self.db is None:
                    self.db = FAISS.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                else:
                    self.db.add_documents(documents)
                
                # Save FAISS index
                os.makedirs(self.persist_directory, exist_ok=True)
                index_path = os.path.join(self.persist_directory, "faiss_index")
                self.db.save_local(index_path)
            
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.db is None:
            return []
        
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents with relevance scores"""
        if self.db is None:
            return []
        
        try:
            return self.db.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error during similarity search with score: {str(e)}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.db = None
            self._initialize_db()
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        info = {
            "store_type": self.store_type,
            "persist_directory": self.persist_directory,
            "exists": self.db is not None
        }
        
        if self.store_type == "chroma" and self.db is not None:
            try:
                info["count"] = self.db._collection.count()
            except:
                info["count"] = "Unknown"
        
        return info