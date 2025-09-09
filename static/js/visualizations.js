// Visualization handling

class NeuralNetworkVisualizer {
    constructor() {
        this.currentTab = 'overview';
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.viz-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.layer);
            });
        });
    }

    switchTab(layerName) {
        // Update active tab
        document.querySelectorAll('.viz-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.layer === layerName);
        });

        // Update active panel
        document.querySelectorAll('.viz-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${layerName}-tab`);
        });

        this.currentTab = layerName;
    }

    updateVisualizations(data) {
        // Show visualization section
        const vizSection = document.getElementById('visualization-section');
        vizSection.classList.remove('hidden');

        // Update prediction display
        document.getElementById('predicted-digit').textContent = data.prediction;
        document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('prediction-result').classList.remove('hidden');

        // Update probability chart
        if (data.visualizations.probability_chart) {
            document.getElementById('prob-chart').src = `data:image/png;base64,${data.visualizations.probability_chart}`;
            document.getElementById('probability-section').classList.remove('hidden');
        }

        // Update layer visualizations
        if (data.visualizations.conv1) {
            document.getElementById('conv1-plot').src = `data:image/png;base64,${data.visualizations.conv1}`;
        }
        if (data.visualizations.conv2) {
            document.getElementById('conv2-plot').src = `data:image/png;base64,${data.visualizations.conv2}`;
        }
        if (data.visualizations.conv3) {
            document.getElementById('conv3-plot').src = `data:image/png;base64,${data.visualizations.conv3}`;
        }
        if (data.visualizations.summary) {
            document.getElementById('summary-plot').src = `data:image/png;base64,${data.visualizations.summary}`;
        }

        // Update layer info
        if (data.layer_info) {
            this.updateLayerInfo(data.layer_info);
        }

        // Scroll to visualizations
        vizSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    updateLayerInfo(layerInfo) {
        // Update conv1 info
        if (layerInfo.conv1) {
            document.getElementById('conv1-filters').textContent = layerInfo.conv1.num_filters || '32';
            if (layerInfo.conv1.shape) {
                document.getElementById('conv1-shape').textContent =
                    `${layerInfo.conv1.shape[0]}×${layerInfo.conv1.shape[1]}`;
            }
        }

        // Update conv2 info
        if (layerInfo.conv2) {
            document.getElementById('conv2-filters').textContent = layerInfo.conv2.num_filters || '64';
            if (layerInfo.conv2.shape) {
                document.getElementById('conv2-shape').textContent =
                    `${layerInfo.conv2.shape[0]}×${layerInfo.conv2.shape[1]}`;
            }
        }

        // Update conv3 info
        if (layerInfo.conv3) {
            document.getElementById('conv3-filters').textContent = layerInfo.conv3.num_filters || '128';
            if (layerInfo.conv3.shape) {
                document.getElementById('conv3-shape').textContent =
                    `${layerInfo.conv3.shape[0]}×${layerInfo.conv3.shape[1]}`;
            }
        }
    }

    showError(message) {
        // Hide visualization section on error
        document.getElementById('visualization-section').classList.add('hidden');
        document.getElementById('prediction-result').classList.add('hidden');
        document.getElementById('probability-section').classList.add('hidden');
    }
}

// Initialize visualizer
const visualizer = new NeuralNetworkVisualizer();

// Update the main.js predict function to use visualizer
// This would be integrated with the existing main.js code
window.updateVisualizationsCallback = (data) => {
    visualizer.updateVisualizations(data);
};

// Tutorial and help functionality
function initializeTutorial() {
    // Show tutorial banner for first-time users
    if (!localStorage.getItem('tutorialDismissed')) {
        document.getElementById('tutorial-banner').classList.remove('hidden');
    }

    // Tutorial button
    document.getElementById('tutorial-btn').addEventListener('click', () => {
        showHelp();
    });

    // Dismiss tutorial
    document.getElementById('dismiss-tutorial').addEventListener('click', () => {
        document.getElementById('tutorial-banner').classList.add('hidden');
        localStorage.setItem('tutorialDismissed', 'true');
    });

    // Help modal
    const modal = document.getElementById('help-modal');
    const closeBtn = modal.querySelector('.close');

    closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.classList.add('hidden');
        }
    });
}

function showHelp() {
    document.getElementById('help-modal').classList.remove('hidden');
}

// Add help button to header
function addHelpButton() {
    const helpBtn = document.createElement('button');
    helpBtn.textContent = '?';
    helpBtn.className = 'help-btn';
    helpBtn.title = 'Help';
    helpBtn.addEventListener('click', showHelp);
    document.body.appendChild(helpBtn);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initializeTutorial();
    addHelpButton();
});
