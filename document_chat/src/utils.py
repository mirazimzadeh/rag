import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
            
        # Find the last period or newline in the chunk
        if end < text_length:
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            split_point = max(last_period, last_newline)
            
            if split_point > start:
                end = split_point + 1
                
        chunks.append(text[start:end])
        start = end - chunk_overlap
        
    return chunks

def create_faiss_index(
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_metric: str = "cosine",
    use_gpu: bool = False
) -> faiss.Index:
    """Create a new FAISS index.
    
    Args:
        embedding_model: Name of the sentence-transformers model
        similarity_metric: Metric to use for similarity search
        use_gpu: Whether to use GPU for embeddings
        
    Returns:
        FAISS index
    """
    try:
        # Initialize the embedding model
        device = 'cuda' if use_gpu else 'cpu'
        model = SentenceTransformer(embedding_model, device=device)
        
        # Get embedding dimension
        dim = model.get_sentence_embedding_dimension()
        
        # Create index based on similarity metric
        if similarity_metric == "cosine":
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        else:
            index = faiss.IndexFlatL2(dim)  # L2 distance
            
        return index
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise

def get_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> List[List[float]]:
    """Get embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
        batch_size: Batch size for processing
        
    Returns:
        List of embedding vectors
    """
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise 