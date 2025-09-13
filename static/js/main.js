/**
 * MNIST Digit Classifier - Main JavaScript
 *
 * This script handles:
 * - Canvas drawing functionality
 * - API communication
 * - Result display
 * - User interactions
 */

// Global variables
let canvas;
let ctx;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCanvas();
    setupEventListeners();
});

/**
 * Initialize the drawing canvas
 */
function initializeCanvas() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Set drawing style
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Fill with white background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Canvas events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    // Button events
    document.getElementById('predict-btn').addEventListener('click', predict);
    document.getElementById('clear-btn').addEventListener('click', clearCanvas);

    // Prevent scrolling when touching canvas
    document.body.addEventListener('touchstart', function(e) {
        if (e.target === canvas) {
            e.preventDefault();
        }
    }, { passive: false });

    document.body.addEventListener('touchend', function(e) {
        if (e.target === canvas) {
            e.preventDefault();
        }
    }, { passive: false });

    document.body.addEventListener('touchmove', function(e) {
        if (e.target === canvas) {
            e.preventDefault();
        }
    }, { passive: false });
}

/**
 * Start drawing on canvas
 */
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getMousePos(e);
}

/**
 * Draw on canvas
 */
function draw(e) {
    if (!isDrawing) return;

    const [currentX, currentY] = getMousePos(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    [lastX, lastY] = [currentX, currentY];
}

/**
 * Stop drawing
 */
function stopDrawing() {
    isDrawing = false;
}

/**
 * Handle touch events
 */
function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' :
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

/**
 * Get mouse position relative to canvas
 */
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return [
        e.clientX - rect.left,
        e.clientY - rect.top
    ];
}

/**
 * Clear the canvas
 */
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Clear results and visualizations
    clearResults();
}

/**
 * Clear prediction results
 */
function clearResults() {
    document.getElementById('probability-section').classList.add('hidden');
    document.getElementById('visualization-section').classList.add('hidden');
}

/**
 * Make prediction
 */
async function predict() {
    // Check if canvas is empty
    if (isCanvasEmpty()) {
        showMessage('Please draw a digit first!', 'error');
        return;
    }

    // Show loading state
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';

    try {
        // Get canvas data as base64
        const imageData = canvas.toDataURL('image/png');

        // Send to API (always use HTML format)
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                image: imageData,
                format: 'html'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update visualizations
        if (window.updateVisualizationsCallback) {
            window.updateVisualizationsCallback(data);
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showMessage('Prediction failed. Please try again.', 'error');

        if (window.visualizer) {
            window.visualizer.showError(error.message);
        }
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
}

/**
 * Check if canvas is empty
 */
function isCanvasEmpty() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Check if all pixels are white
    for (let i = 0; i < data.length; i += 4) {
        // Check RGB values (ignore alpha)
        if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
            return false;
        }
    }
    return true;
}

/**
 * Show a temporary message
 */
function showMessage(message, type = 'info') {
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `message message-${type}`;
    messageEl.textContent = message;

    // Add to page
    document.body.appendChild(messageEl);

    // Remove after 3 seconds
    setTimeout(() => {
        messageEl.remove();
    }, 3000);
}

// Add message styles
const style = document.createElement('style');
style.textContent = `
    .message {
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        border-radius: 5px;
        color: white;
        font-weight: 500;
        z-index: 1001;
        animation: slideDown 0.3s ease-out;
    }

    .message-error {
        background-color: #dc3545;
    }

    .message-info {
        background-color: #17a2b8;
    }

    @keyframes slideDown {
        from {
            transform: translateX(-50%) translateY(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);
