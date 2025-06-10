import os
import fitz  # PyMuPDF
import docx  # python-docx
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from typing import List, Set, Optional, Generator

console = Console()

class DocumentProcessor:
    """
    Handles document scanning and text extraction from various file formats.
    """
    def __init__(self, folder_path: str, extensions: Optional[Set[str]] = None):
        """
        Initialize document processor with folder path and allowed extensions.
        
        Args:
            folder_path: Path to the folder containing documents
            extensions: Set of allowed file extensions (default: {'.pdf', '.docx'})
        """
        self.folder_path = Path(folder_path)
        self.extensions = extensions or {'.pdf', '.docx'}

    def _get_all_files(self) -> List[Path]:
        """
        Recursively find all files in folder_path matching the allowed extensions.
        
        Returns:
            List of file paths
        """
        matched_files = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if Path(file).suffix.lower() in self.extensions:
                    matched_files.append(Path(root) / file)
        return matched_files

    def scan_documents(self) -> Generator[Path, None, None]:
        """
        Generator that yields files one by one while showing a live progress bar.
        
        Yields:
            Path objects for each matching file
        """
        files = self._get_all_files()
        total = len(files)

        with Progress(
            TextColumn("[bold cyan]Processing file {task.completed}/{task.total}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Scanning documents...", total=total)
            for file_path in files:
                yield file_path
                progress.update(task, advance=1)

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text or empty string on failure
        """
        try:
            doc = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            console.log(f"[red]❌ Error reading PDF {file_path}: {e}")
            return ""

    def _extract_text_from_docx(self, file_path: Path) -> str:
        """
        Extract text from DOCX using python-docx.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text or empty string on failure
        """
        try:
            doc = docx.Document(file_path)
            text = '\n'.join(p.text for p in doc.paragraphs)
            return text
        except Exception as e:
            console.log(f"[red]❌ Error reading DOCX {file_path}: {e}")
            return ""

    def extract_text(self, file_path: Path) -> str:
        """
        Extract text based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text or empty string on failure
        """
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self._extract_text_from_docx(file_path)
        else:
            console.log(f"[yellow]⚠️ Unsupported file type: {ext}")
            return "" 