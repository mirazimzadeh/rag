import pickle
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from rich.console import Console

from .utils import SearchResult, get_text_splitter, create_faiss_index

console = Console()

class VectorDB:
    def __init__(self,
                 vector_dir: str = "vector",
                 vector_file: str = "vector.index",
                 texts_file: str = "texts.pkl",
                 metadata_file: str = "metadata.json",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embed_model_name: str = "all-mpnet-base-v2",
                 similarity_metric: str = "cosine",
                 normalize_embeddings: bool = True,
                 preserve_metadata: bool = True,
                 use_gpu: bool = False):
        """
        Enhanced VectorDB with better chunking, metadata support, and optimization.
        """
        self.vector_dir = Path(vector_dir)
        self.vector_file = self.vector_dir / vector_file
        self.texts_file = self.vector_dir / texts_file
        self.metadata_file = self.vector_dir / metadata_file
       
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name
        self.similarity_metric = similarity_metric
        self.normalize_embeddings = normalize_embeddings
        self.preserve_metadata = preserve_metadata
        self.use_gpu = use_gpu
       
        # Storage
        self.index = None
        self.texts = []
        self.metadata = []
       
        # Setup
        self._setup_logging()
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        device = 'cuda' if use_gpu else 'cpu'
        self.embedder = SentenceTransformer(self.embed_model_name, device=device)
        self._load_vector_db()

    def _setup_logging(self):
        """Setup logging for better debugging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _split_text_enhanced(self, texts: Union[str, List[str]],
                           text_type: str = "general",
                           source_metadata: Optional[List[Dict]] = None):
        """
        Enhanced text splitting with metadata preservation.
        """
        console.log("[blue]Splitting text into chunks with enhanced processing...")
       
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

    def _load_vector_db(self):
        """Load existing vector database"""
        try:
            if (self.vector_file.exists() and
                self.texts_file.exists() and
                self.metadata_file.exists()):
               
                console.log("[green]Loading existing enhanced vector DB...")
               
                self.index = faiss.read_index(str(self.vector_file))
               
                with open(self.texts_file, "rb") as f:
                    self.texts = pickle.load(f)
               
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
               
                console.log(f"Loaded {len(self.texts)} texts with metadata; "
                          f"index dimension: {self.index.d}")
            else:
                console.log("[yellow]No existing enhanced vector DB found. Starting fresh.")
                self._initialize_empty_db()
               
        except Exception as e:
            console.log(f"[red]Error loading vector DB: {e}")
            self._initialize_empty_db()

    def _initialize_empty_db(self):
        """Initialize empty database"""
        self.index = None
        self.texts = []
        self.metadata = []

    def _save_vector_db(self):
        """Save vector database"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.vector_file))
               
                with open(self.texts_file, "wb") as f:
                    pickle.dump(self.texts, f)
               
                with open(self.metadata_file, "w") as f:
                    json.dump(self.metadata, f, indent=2)
               
                console.log(f"[green]Saved enhanced vector DB: {len(self.texts)} texts with metadata")
            else:
                console.log("[yellow]No index available to save.")
        except Exception as e:
            console.log(f"[red]Error saving vector DB: {e}")

    def add(self, new_texts: Union[str, List[str]],
            text_type: str = "general",
            source_metadata: Optional[List[Dict]] = None):
        """Add new texts to the vector database"""
        if not new_texts:
            console.log("[yellow]No new texts provided to add.")
            return

        if not isinstance(new_texts, list):
            new_texts = [new_texts]

        console.log(f"[blue]Adding {len(new_texts)} new texts with enhanced processing...")

        try:
            docs, chunk_metadata = self._split_text_enhanced(
                new_texts, text_type, source_metadata
            )
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

            self._save_vector_db()
            console.log(f"[green]Successfully added {len(new_chunks)} chunks to enhanced vector DB")
           
        except Exception as e:
            console.log(f"[red]Error adding texts to vector DB: {e}")
            self.logger.error(f"Add operation failed: {e}", exc_info=True)

    def search(self, query: str,
               top_k: int = 10,
               similarity_threshold: float = 0.0,
               return_metadata: bool = True) -> List[SearchResult]:
        """Search the vector database"""
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
                        metadata = self.metadata[idx] if return_metadata and idx < len(self.metadata) else {}
                       
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
        """Get database statistics"""
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