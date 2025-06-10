# Document Chat

A powerful document chat application that combines document processing, vector storage, and LLM-based chat capabilities. This application allows you to upload documents, process them, and chat with them using advanced RAG (Retrieval-Augmented Generation) techniques.

## Features

- **Document Processing**

  - Support for PDF, DOCX, and TXT files
  - Automatic text extraction and chunking
  - Metadata preservation

- **Vector Storage**

  - FAISS-based vector database
  - Efficient similarity search
  - Configurable chunking parameters

- **Chat Interface**

  - Modern web UI
  - Real-time chat with documents
  - RAG controls for fine-tuning responses
  - Document source tracking

- **LLM Integration**
  - Ollama model support
  - Configurable model parameters
  - Context-aware responses

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd document-chat
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama:

Follow the instructions at [Ollama's website](https://ollama.ai) to install Ollama on your system.

5. Pull the required model:

```bash
ollama pull gemma:2b-it-qat
```

## Usage

1. Start the FastAPI server:

```bash
python -m uvicorn document_chat.api:app --reload --host 0.0.0.0 --port 8000
```

2. Open your browser and navigate to:

```
http://localhost:8000
```

3. Upload documents using the web interface

4. Start chatting with your documents!

## Configuration

The application can be configured through the `config.py` file:

- **Ollama Settings**

  - Model selection
  - API configuration
  - Model parameters

- **Vector Store Settings**

  - Chunk size and overlap
  - Embedding model
  - Similarity metrics

- **RAG Settings**
  - Maximum context chunks
  - Similarity thresholds
  - Model selection

## API Endpoints

- `POST /upload`: Upload and process documents
- `POST /chat`: Send a message and get a response
- `GET /stats`: Get system statistics

## Project Structure

```
document_chat/
├── api.py              # FastAPI application
├── config.py           # Configuration settings
├── core/
│   ├── document_processor.py  # Document processing
│   ├── vector_store.py        # Vector storage
│   └── llm_client.py          # LLM integration
├── static/
│   ├── index.html     # Web interface
│   ├── styles.css     # Styling
│   └── script.js      # Frontend logic
└── utils/
    └── helpers.py     # Utility functions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://github.com/ollama/ollama) for the LLM integration
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for text embeddings
