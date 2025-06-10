// DOM Elements
const fileInput = document.getElementById('file-input');
const folderInput = document.getElementById('folder-input');
const fileList = document.getElementById('file-list');
const folderList = document.getElementById('folder-list');
const uploadButton = document.getElementById('upload-button');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const useRagCheckbox = document.getElementById('use-rag');
const ragParamsDiv = document.getElementById('rag-params');
const maxChunksInput = document.getElementById('max-chunks');
const similarityThresholdInput = document.getElementById('similarity-threshold');
const similarityValueDisplay = document.getElementById('similarity-value');
const docCount = document.getElementById('doc-count');
const chunkCount = document.getElementById('chunk-count');

// State
let isProcessing = false;
let selectedFiles = new Set();
let selectedFolders = new Set();
let defaultModel = 'gemma:2b-it-qat'; // Will be updated from config

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
folderInput.addEventListener('change', handleFolderSelect);
uploadButton.addEventListener('click', handleUpload);
sendButton.addEventListener('click', handleSend);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

useRagCheckbox.addEventListener('change', () => {
    ragParamsDiv.style.display = useRagCheckbox.checked ? 'grid' : 'none';
});

similarityThresholdInput.addEventListener('input', () => {
    const value = similarityThresholdInput.value / 100;
    similarityValueDisplay.textContent = value.toFixed(2);
});

// File Selection Handlers
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    files.forEach(file => {
        if (isValidFile(file)) {
            selectedFiles.add(file);
        }
    });
    updateFileList();
}

function handleFolderSelect(event) {
    const files = Array.from(event.target.files);
    const folder = files[0]?.webkitRelativePath.split('/')[0];
    if (folder) {
        selectedFolders.add(folder);
        files.forEach(file => {
            if (isValidFile(file)) {
                selectedFiles.add(file);
            }
        });
    }
    updateFileList();
}

function isValidFile(file) {
    const validTypes = ['.pdf', '.docx', '.txt'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    return validTypes.includes(extension);
}

function updateFileList() {
    const fileNames = Array.from(selectedFiles).map(file => file.name);
    const folderNames = Array.from(selectedFolders);

    fileList.textContent = fileNames.length > 0
        ? `Selected files: ${fileNames.join(', ')}`
        : '';

    folderList.textContent = folderNames.length > 0
        ? `Selected folders: ${folderNames.join(', ')}`
        : '';
}

// Upload Handler
async function handleUpload() {
    if (isProcessing || selectedFiles.size === 0) return;

    try {
        setLoading(true);
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        showSuccess('Documents processed successfully');
        updateStats();

        // Clear selections
        selectedFiles.clear();
        selectedFolders.clear();
        fileInput.value = '';
        folderInput.value = '';
        updateFileList();
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// Chat Handlers
async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    try {
        setLoading(true);
        addMessage(message, 'user');
        messageInput.value = '';

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                rag_params: useRagCheckbox.checked ? {
                    max_chunks: parseInt(maxChunksInput.value),
                    similarity_threshold: parseFloat(similarityThresholdInput.value) / 100,
                    model: defaultModel
                } : null
            })
        });

        if (!response.ok) {
            throw new Error(`Chat failed: ${response.statusText}`);
        }

        const result = await response.json();
        addMessage(result.message.content, 'assistant', result.message.metadata);
        updateStats();
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// UI Helpers
function addMessage(content, type, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);

    if (metadata) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        metadataDiv.textContent = `Source: ${metadata.source || 'Unknown'}`;
        messageDiv.appendChild(metadataDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function updateStats() {
    try {
        const response = await fetch('/stats');
        if (!response.ok) throw new Error('Failed to fetch stats');

        const stats = await response.json();
        docCount.textContent = stats.vector_store_stats.document_count || 0;
        chunkCount.textContent = stats.vector_store_stats.chunk_count || 0;
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatMessages.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;
    chatMessages.appendChild(successDiv);
    setTimeout(() => successDiv.remove(), 5000);
}

function setLoading(loading) {
    isProcessing = loading;
    uploadButton.disabled = loading;
    sendButton.disabled = loading;
    messageInput.disabled = loading;
    document.body.classList.toggle('loading', loading);
}

// Initialize
updateStats(); 