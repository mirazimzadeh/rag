import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from .utils import split_text, create_faiss_index, get_embeddings

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages document storage and retrieval using FAISS vector database."""
    
    def __init__(
        self,
        store_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_metric: str = "cosine",
        use_gpu: bool = False
    ):
        """Initialize the vector store.
        
        Args:
            store_dir: Directory to store the vector database
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the sentence-transformers model to use
            similarity_metric: Metric to use for similarity search
            use_gpu: Whether to use GPU for embeddings
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self.use_gpu = use_gpu
        
        self.index = None
        self.metadata = []
        self.model = None
        self._load_or_create_index()
        
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        index_path = self.store_dir / "index.faiss"
        metadata_path = self.store_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded existing index with {len(self.metadata)} documents")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
            
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = create_faiss_index(
            self.embedding_model,
            self.similarity_metric,
            self.use_gpu
        )
        self.model = SentenceTransformer(
            self.embedding_model,
            device='cuda' if self.use_gpu else 'cpu'
        )
        self.metadata = []
        self._save_index()
        
    def _save_index(self):
        """Save the current index and metadata."""
        try:
            faiss.write_index(self.index, str(self.store_dir / "index.faiss"))
            with open(self.store_dir / "metadata.json", 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of dictionaries containing document text and metadata
        """
        if not documents:
            return
            
        # Split documents into chunks
        chunks = []
        for doc in documents:
            doc_chunks = split_text(
                doc["text"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            for chunk in doc_chunks:
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": len(chunks)
                    }
                })
                
        if not chunks:
            return
            
        # Create embeddings for chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts, self.model)
        
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Update metadata
        self.metadata.extend(chunks)
        
        # Save changes
        self._save_index()
        
    def search(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of dictionaries containing matched chunks and their metadata
        """
        if not self.metadata:
            return []
            
        # Get query embedding
        query_embedding = get_embeddings([query], self.model)[0]
        
        # Search index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            k
        )
        
        # Filter and format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and distance >= similarity_threshold:
                chunk = self.metadata[idx]
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "similarity": float(distance)
                })
                
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "document_count": len(set(m["metadata"]["filename"] for m in self.metadata)),
            "chunk_count": len(self.metadata),
            "embedding_model": self.embedding_model,
            "similarity_metric": self.similarity_metric
        } 