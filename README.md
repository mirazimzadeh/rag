# Document Chat

A document-based chat application that allows you to upload documents and chat with them using Ollama and FAISS for semantic search.

## Features

- Upload and process PDF and DOCX documents
- Extract text and create semantic embeddings
- Store document chunks in a FAISS vector database
- Chat with documents using Ollama LLM
- Modern web interface for document upload and chat
- Real-time statistics and monitoring

## Requirements

- Python 3.8 or higher
- Ollama installed and running locally (or accessible via API)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/document-chat.git
cd document-chat
```

2. Install the package:

```bash
pip install -e .
```

## Usage

1. Start the application:

```bash
document-chat
```

2. Open your browser and navigate to `http://localhost:8000`

3. Upload your documents using the web interface

4. Start chatting with your documents!

## API Endpoints

- `GET /`: Web interface
- `POST /upload`: Upload and process documents
- `POST /chat`: Send a chat message
- `GET /stats`: Get system statistics
- `GET /models`: List available Ollama models

## Development

### Project Structure

```
document_chat/
├── core/
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── llm_client.py
│   └── chat_manager.py
├── static/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── api.py
└── __init__.py
```

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [Ollama](https://github.com/ollama/ollama) for the LLM integration
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for text embeddings
