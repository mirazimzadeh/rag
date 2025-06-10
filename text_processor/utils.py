from dataclasses import dataclass
from typing import Dict, Any, List
from rich.console import Console

console = Console()

@dataclass
class SearchResult:
    """Structure for search results with metadata"""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: int

def get_text_splitter(chunk_size: int, chunk_overlap: int, text_type: str = "general"):
    """
    Get appropriate text splitter based on content type.
    Enhanced with better separators and sentence boundary respect.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

    if text_type == "markdown":
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
   
    # Enhanced separators for better chunking
    separators = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ".",     # Sentences
        "!",     # Exclamations
        "?",     # Questions
        ";",     # Semicolons
        ",",     # Commas
        " ",     # Words
        ""       # Characters
    ]
   
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
        is_separator_regex=False
    )

def create_faiss_index(dimension: int, similarity_metric: str = "cosine", use_gpu: bool = False):
    """
    Create optimized FAISS index based on similarity metric.
    """
    import faiss

    if similarity_metric == "cosine":
        # Use inner product with normalized vectors for cosine similarity
        index = faiss.IndexFlatIP(dimension)
    elif similarity_metric == "inner_product":
        index = faiss.IndexFlatIP(dimension)
    else:  # l2 (default)
        index = faiss.IndexFlatL2(dimension)
   
    # Use GPU if available and requested
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        console.log("[green]Using GPU for FAISS operations")
   
    return index 