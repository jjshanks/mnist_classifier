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
    canvas = document.getElementById('drawingCanvas');
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
    document.getElementById('predictBtn').addEventListener('click', predict);
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);

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

    // Hide the "Draw here" label
    document.querySelector('.canvas-label').style.opacity = '0';
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

    // Show the "Draw here" label again
    document.querySelector('.canvas-label').style.opacity = '1';

    // Clear results
    clearResults();
}

/**
 * Clear prediction results
 */
function clearResults() {
    document.getElementById('resultContainer').innerHTML = `
        <div class="placeholder">
            <div class="placeholder-icon">✏️</div>
            <p>Draw a digit and click "Predict"</p>
        </div>
    `;
    document.getElementById('probabilityChart').style.display = 'none';
}

/**
 * Make prediction
 */
async function predict() {
    // Check if canvas is empty
    if (isCanvasEmpty()) {
        showError('Please draw a digit first!');
        return;
    }

    // Show loading overlay
    showLoading();

    try {
        // Get canvas data as base64
        const imageData = canvas.toDataURL('image/png');

        // Send to API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Hide loading
        hideLoading();

        // Display results
        displayResults(result);

    } catch (error) {
        hideLoading();
        showError('Failed to make prediction. Please try again.');
        console.error('Prediction error:', error);
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
 * Display prediction results
 */
function displayResults(result) {
    // Display main prediction
    document.getElementById('resultContainer').innerHTML = `
        <div class="prediction-result">
            <div class="predicted-digit">${result.predicted_digit}</div>
            <div class="confidence-score">${(result.confidence * 100).toFixed(1)}% confident</div>
            <div class="processing-time">Processed in ${result.processing_time.toFixed(1)}ms</div>
        </div>
    `;

    // Display probability chart
    displayProbabilityChart(result.probabilities);
}

/**
 * Display probability chart
 */
function displayProbabilityChart(probabilities) {
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.innerHTML = '';

    // Sort probabilities by value
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    // Create bars for each digit
    sortedProbs.forEach(([digit, probability]) => {
        const percentage = (probability * 100).toFixed(1);
        const isHigh = probability > 0.5;

        const probBar = document.createElement('div');
        probBar.className = 'prob-bar';
        probBar.innerHTML = `
            <div class="prob-label">${digit}</div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill ${isHigh ? 'high' : ''}" style="width: ${percentage}%"></div>
            </div>
            <div class="prob-value">${percentage}%</div>
        `;

        chartContainer.appendChild(probBar);
    });

    // Show the chart
    document.getElementById('probabilityChart').style.display = 'block';
}

/**
 * Show loading overlay
 */
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorModal').style.display = 'flex';
}

/**
 * Close error modal
 */
function closeErrorModal() {
    document.getElementById('errorModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('errorModal');
    if (event.target === modal) {
        closeErrorModal();
    }
}

// Close modal with close button
document.addEventListener('DOMContentLoaded', function() {
    const closeBtn = document.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = closeErrorModal;
    }
});
