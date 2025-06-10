import os
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document scanning and text extraction from various file formats."""
    
    def __init__(self, upload_dir: str):
        """Initialize the document processor.
        
        Args:
            upload_dir: Directory where documents will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all files in a directory recursively."""
        files = []
        for item in directory.iterdir():
            if item.is_file():
                files.append(item)
            elif item.is_dir():
                files.extend(self._get_all_files(item))
        return files
    
    def scan_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Scan all documents in a directory and extract their text.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of dictionaries containing document metadata and text
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory {directory} does not exist")
            
        results = []
        files = self._get_all_files(directory)
        
        for file in files:
            try:
                text = self.extract_text(file)
                if text:
                    results.append({
                        "path": str(file),
                        "text": text,
                        "metadata": {
                            "filename": file.name,
                            "extension": file.suffix.lower(),
                            "size": file.stat().st_size
                        }
                    })
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                continue
                
        return results
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text or None if extraction fails
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif extension == '.docx':
            return self._extract_text_from_docx(file_path)
        elif extension == '.txt':
            try:
                return file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {str(e)}")
                return None
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return None 