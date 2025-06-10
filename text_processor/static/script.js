// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const modelSelect = document.getElementById('modelSelect');
const statsContent = document.getElementById('statsContent');

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
chatInput.addEventListener('input', handleInputChange);
chatInput.addEventListener('keydown', handleKeyPress);
sendBtn.addEventListener('click', sendMessage);
modelSelect.addEventListener('change', updateStats);

// Auto-resize textarea
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Functions
async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files.length) return;

    uploadStatus.innerHTML = '<span class="loading"></span> Processing documents...';
    uploadBtn.disabled = true;

    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        const response = await fetch(`${API_BASE_URL}/process-folder`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        uploadStatus.textContent = 'Documents processed successfully!';
        sendBtn.disabled = false;
        updateStats();
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Error processing documents. Please try again.';
    } finally {
        uploadBtn.disabled = false;
    }
}

function handleInputChange() {
    sendBtn.disabled = !chatInput.value.trim();
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!sendBtn.disabled) {
            sendMessage();
        }
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    try {
        // Show loading message
        const loadingId = addMessage('Thinking...', 'system');

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                model: modelSelect.value
            })
        });

        if (!response.ok) throw new Error('Chat request failed');

        const result = await response.json();

        // Remove loading message
        document.getElementById(loadingId).remove();

        // Add assistant's response
        addMessage(result.response, 'assistant');

        // Add sources if available
        if (result.sources && result.sources.length > 0) {
            const sourcesText = result.sources
                .map((source, i) => `Source ${i + 1} (Score: ${source.score.toFixed(2)}):\n${source.text}`)
                .join('\n\n');
            addMessage(sourcesText, 'system');
        }
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('Sorry, there was an error processing your request.', 'system');
    }
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    const id = 'msg-' + Date.now();
    messageDiv.id = id;
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');

        const stats = await response.json();

        if (stats.status === 'empty') {
            statsContent.innerHTML = 'No documents processed yet.';
            return;
        }

        statsContent.innerHTML = `
            <p>Total Chunks: ${stats.total_chunks}</p>
            <p>Model: ${stats.embedding_model}</p>
            <p>Chunk Size: ${stats.chunk_size}</p>
            <p>Current LLM: ${modelSelect.value}</p>
        `;
    } catch (error) {
        console.error('Stats error:', error);
        statsContent.innerHTML = 'Error loading statistics.';
    }
}

// Initial stats update
updateStats(); 