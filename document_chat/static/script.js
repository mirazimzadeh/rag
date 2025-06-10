// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const statsContent = document.getElementById('statsContent');
const vectorStatsContent = document.getElementById('vectorStatsContent');

// RAG Controls
const useRagCheckbox = document.getElementById('useRag');
const ragParamsDiv = document.getElementById('ragParams');
const maxChunksInput = document.getElementById('maxChunks');
const similarityThresholdInput = document.getElementById('similarityThreshold');
const similarityValueDisplay = document.getElementById('similarityValue');
const modelSelect = document.getElementById('modelSelect');

// State
let chatHistory = [];
let isProcessing = false;

// Event Listeners
uploadBtn.addEventListener('click', handleUpload);
sendBtn.addEventListener('click', handleSend);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// RAG Control Event Listeners
useRagCheckbox.addEventListener('change', (e) => {
    ragParamsDiv.classList.toggle('hidden', !e.target.checked);
});

similarityThresholdInput.addEventListener('input', (e) => {
    const value = e.target.value / 100;
    similarityValueDisplay.textContent = value.toFixed(2);
});

// Error Handling
function showError(element, message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    element.classList.add('input-error');
    element.parentNode.insertBefore(errorDiv, element.nextSibling);
    setTimeout(() => {
        errorDiv.remove();
        element.classList.remove('input-error');
    }, 5000);
}

function setLoading(isLoading) {
    isProcessing = isLoading;
    document.body.classList.toggle('loading', isLoading);
    sendBtn.disabled = isLoading;
    uploadBtn.disabled = isLoading;
}

// Functions
async function handleUpload() {
    if (isProcessing) return;

    const files = fileInput.files;
    if (files.length === 0) {
        showError(fileInput, 'Please select files to upload');
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    try {
        setLoading(true);
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();
        addMessage('assistant', `Successfully processed ${result.message}`);
        updateStats();
    } catch (error) {
        console.error('Upload error:', error);
        showError(uploadBtn, error.message || 'Error uploading files. Please try again.');
    } finally {
        setLoading(false);
        fileInput.value = '';
    }
}

async function handleSend() {
    if (isProcessing) return;

    const message = messageInput.value.trim();
    if (!message) {
        showError(messageInput, 'Please enter a message');
        return;
    }

    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';

    try {
        setLoading(true);
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                history: chatHistory,
                rag_params: useRagCheckbox.checked ? {
                    max_chunks: parseInt(maxChunksInput.value),
                    similarity_threshold: parseFloat(similarityThresholdInput.value) / 100,
                    model: modelSelect.value
                } : null
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Chat request failed');
        }

        const result = await response.json();
        addMessage('assistant', result.message.content, result.message.metadata);
        updateStats();
    } catch (error) {
        console.error('Chat error:', error);
        showError(sendBtn, error.message || 'Error processing your message. Please try again.');
    } finally {
        setLoading(false);
    }
}

function addMessage(role, content, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);

    if (metadata) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        metadataDiv.textContent = `Model: ${metadata.model} | Context chunks: ${metadata.context_chunks}`;
        messageDiv.appendChild(metadataDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Update chat history
    chatHistory.push({
        role,
        content,
        metadata
    });
}

async function updateStats() {
    try {
        const response = await fetch('/stats');
        if (!response.ok) {
            throw new Error('Failed to fetch stats');
        }

        const stats = await response.json();
        displayStats(stats);
    } catch (error) {
        console.error('Stats error:', error);
        showError(statsContent, 'Error updating statistics');
    }
}

function displayStats(stats) {
    const { chat_stats, vector_store_stats } = stats;

    // Display chat stats
    let chatStatsHtml = `
        <div class="stat-group">
            <h3>Chat Stats</h3>
            <p>Model: ${chat_stats.model}</p>
            <p>Max Context Chunks: ${chat_stats.max_context_chunks}</p>
            <p>Similarity Threshold: ${chat_stats.similarity_threshold}</p>
        </div>
    `;
    statsContent.innerHTML = chatStatsHtml;

    // Display vector store stats
    let vectorStatsHtml = `
        <div class="stat-group">
            <h3>Vector Store Stats</h3>
            <p>Total Chunks: ${vector_store_stats.total_chunks}</p>
            <p>Index Dimension: ${vector_store_stats.index_dimension}</p>
            <p>Embedding Model: ${vector_store_stats.embedding_model}</p>
            <p>Chunk Size: ${vector_store_stats.chunk_size}</p>
            <p>Chunk Overlap: ${vector_store_stats.chunk_overlap}</p>
            <p>Similarity Metric: ${vector_store_stats.similarity_metric}</p>
            <p>Has Metadata: ${vector_store_stats.has_metadata ? 'Yes' : 'No'}</p>
            <p>GPU Enabled: ${vector_store_stats.use_gpu ? 'Yes' : 'No'}</p>
        </div>
    `;
    vectorStatsContent.innerHTML = vectorStatsHtml;
}

// Initial setup
updateStats(); 