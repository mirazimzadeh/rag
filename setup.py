from setuptools import setup, find_packages

setup(
    name="document_chat",
    version="0.1.0",
    description="A powerful document chat application with RAG capabilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "PyMuPDF>=1.19.0",
        "python-docx>=0.8.11",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "httpx>=0.24.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "document-chat=document_chat.src.api:app"
        ]
    }
) 