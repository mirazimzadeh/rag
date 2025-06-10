from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="document-chat",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A document-based chat application using Ollama and FAISS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/document-chat",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "PyMuPDF>=1.19.0",
        "python-docx>=0.8.11",
        "rich>=10.0.0",
        "httpx>=0.23.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "document-chat=document_chat.api:main",
        ],
    },
) 