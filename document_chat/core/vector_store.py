import pickle
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from rich.console import Console

from ..utils.helpers import get_text_splitter, create_faiss_index

console = Console()

@dataclass
class SearchResult:
    """Structure for search results with metadata"""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: int

class VectorStore:
    """
    Manages vector storage and retrieval for document chunks.
    """
    def __init__(self,
                 store_dir: str = "vector_store",
                 index_file: str = "index.faiss",
                 texts_file: str = "texts.pkl",
                 metadata_file: str = "metadata.json",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embed_model_name: str = "all-mpnet-base-v2",
                 similarity_metric: str = "cosine",
                 normalize_embeddings: bool = True,
                 use_gpu: bool = False):
        """
        Initialize vector store with configuration.
        
        Args:
            store_dir: Directory to store vector files
            index_file: FAISS index filename
            texts_file: Pickled texts filename
            metadata_file: JSON metadata filename
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embed_model_name: Embedding model name
            similarity_metric: Similarity calculation method
            normalize_embeddings: Whether to normalize embeddings
            use_gpu: Use GPU for FAISS operations
        """
        self.store_dir = Path(store_dir)
        self.index_file = self.store_dir / index_file
        self.texts_file = self.store_dir / texts_file
        self.metadata_file = self.store_dir / metadata_file
       
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name
        self.similarity_metric = similarity_metric
        self.normalize_embeddings = normalize_embeddings
        self.use_gpu = use_gpu
       
        # Storage
        self.index = None
        self.texts = []
        self.metadata = []
       
        # Setup
        self._setup_logging()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        device = 'cuda' if use_gpu else 'cpu'
        self.embedder = SentenceTransformer(self.embed_model_name, device=device)
        self._load_store()

    def _setup_logging(self):
        """Setup logging for better debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _split_text(self, texts: Union[str, List[str]],
                   text_type: str = "general",
                   source_metadata: Optional[List[Dict]] = None):
        """
        Split text into chunks with metadata preservation.
        
        Args:
            texts: Text or list of texts to split
            text_type: Type of text for appropriate splitting
            source_metadata: Optional metadata for each text
            
        Returns:
            Tuple of (documents, metadata)
        """
        console.log("[blue]Splitting text into chunks...")
       
        splitter = get_text_splitter(self.chunk_size, self.chunk_overlap, text_type)
       
        if isinstance(texts, str):
            texts = [texts]
       
        all_docs = []
        all_metadata = []
       
        for i, text in enumerate(texts):
            docs = splitter.create_documents([text])
           
            for j, doc in enumerate(docs):
                metadata = {
                    "source_index": i,
                    "chunk_index": j,
                    "chunk_size": len(doc.page_content),
                    "text_type": text_type,
                    "total_chunks_in_source": len(docs)
                }
               
                if source_metadata and i < len(source_metadata):
                    metadata.update(source_metadata[i])
               
                all_docs.append(doc)
                all_metadata.append(metadata)
       
        return all_docs, all_metadata

    def _load_store(self):
        """Load existing vector store"""
        try:
            if (self.index_file.exists() and
                self.texts_file.exists() and
                self.metadata_file.exists()):
               
                console.log("[green]Loading existing vector store...")
               
                self.index = faiss.read_index(str(self.index_file))
               
                with open(self.texts_file, "rb") as f:
                    self.texts = pickle.load(f)
               
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
               
                console.log(f"Loaded {len(self.texts)} chunks; index dimension: {self.index.d}")
            else:
                console.log("[yellow]No existing vector store found. Starting fresh.")
                self._initialize_empty_store()
               
        except Exception as e:
            console.log(f"[red]Error loading vector store: {e}")
            self._initialize_empty_store()

    def _initialize_empty_store(self):
        """Initialize empty store"""
        self.index = None
        self.texts = []
        self.metadata = []

    def _save_store(self):
        """Save vector store"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
               
                with open(self.texts_file, "wb") as f:
                    pickle.dump(self.texts, f)
               
                with open(self.metadata_file, "w") as f:
                    json.dump(self.metadata, f, indent=2)
               
                console.log(f"[green]Saved vector store: {len(self.texts)} chunks")
            else:
                console.log("[yellow]No index available to save.")
        except Exception as e:
            console.log(f"[red]Error saving vector store: {e}")

    def add_documents(self, texts: Union[str, List[str]],
                     text_type: str = "general",
                     source_metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: Text or list of texts to add
            text_type: Type of text for appropriate splitting
            source_metadata: Optional metadata for each text
        """
        if not texts:
            console.log("[yellow]No texts provided to add.")
            return

        if not isinstance(texts, list):
            texts = [texts]

        console.log(f"[blue]Adding {len(texts)} texts...")

        try:
            docs, chunk_metadata = self._split_text(texts, text_type, source_metadata)
            new_chunks = [doc.page_content for doc in docs]

            if not new_chunks:
                console.log("[yellow]No chunks created from texts.")
                return

            console.log("[blue]Generating embeddings...")
            new_embeddings = self.embedder.encode(
                new_chunks,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=self.normalize_embeddings
            )
            new_embeddings_np = np.array(new_embeddings).astype('float32')

            if self.index is None:
                dim = new_embeddings_np.shape[1]
                self.index = create_faiss_index(dim, self.similarity_metric, self.use_gpu)
                self.texts = []
                self.metadata = []

            self.index.add(new_embeddings_np)
            self.texts.extend(new_chunks)
            self.metadata.extend(chunk_metadata)

            self._save_store()
            console.log(f"[green]Successfully added {len(new_chunks)} chunks")
           
        except Exception as e:
            console.log(f"[red]Error adding texts: {e}")
            self.logger.error(f"Add operation failed: {e}", exc_info=True)

    def search(self, query: str,
               top_k: int = 10,
               similarity_threshold: float = 0.0) -> List[SearchResult]:
        """
        Search the vector store.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.texts) == 0:
            console.log("[yellow]No vectors available for search.")
            return []

        try:
            query_embedding = self.embedder.encode(
                [query],
                normalize_embeddings=self.normalize_embeddings
            )
            query_vector = np.array(query_embedding).astype('float32')

            scores, indices = self.index.search(query_vector, top_k)
           
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.texts):
                    if self.similarity_metric in ["cosine", "inner_product"]:
                        similarity_score = float(score)
                    else:
                        similarity_score = 1.0 / (1.0 + float(score))
                   
                    if similarity_score >= similarity_threshold:
                        metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                       
                        results.append(SearchResult(
                            text=self.texts[idx],
                            score=similarity_score,
                            metadata=metadata,
                            chunk_id=idx
                        ))

            console.log(f"[green]Found {len(results)} results above threshold {similarity_threshold}")
            return results
           
        except Exception as e:
            console.log(f"[red]Error during search: {e}")
            self.logger.error(f"Search operation failed: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get store statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.index is None:
            return {"status": "empty"}
       
        return {
            "total_chunks": len(self.texts),
            "index_dimension": self.index.d,
            "embedding_model": self.embed_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_metric": self.similarity_metric,
            "has_metadata": len(self.metadata) > 0,
            "use_gpu": self.use_gpu
        } 