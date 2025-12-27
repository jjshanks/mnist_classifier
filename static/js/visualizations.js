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
        if (vizSection) {
            vizSection.classList.remove('hidden');
        }

        // Update probability chart
        if (data.visualizations && data.visualizations.probability_chart) {
            const probChart = document.getElementById('prob-chart');
            const probSection = document.getElementById('probability-section');
            if (probChart) {
                probChart.src = `data:image/png;base64,${data.visualizations.probability_chart}`;
            }
            if (probSection) {
                probSection.classList.remove('hidden');
            }
        }

        // Update layer visualizations (HTML format only)
        if (data.visualizations) {
            if (data.visualizations.conv1) {
                const conv1Container = document.getElementById('conv1-tab');
                if (conv1Container) {
                    conv1Container.innerHTML = `
                        <h3>First Convolutional Layer</h3>
                        ${data.visualizations.conv1}
                    `;
                }
            }
            
            if (data.visualizations.conv2) {
                const conv2Container = document.getElementById('conv2-tab');
                if (conv2Container) {
                    conv2Container.innerHTML = `
                        <h3>Second Convolutional Layer</h3>
                        ${data.visualizations.conv2}
                    `;
                }
            }
            
            if (data.visualizations.conv3) {
                const conv3Container = document.getElementById('conv3-tab');
                if (conv3Container) {
                    conv3Container.innerHTML = `
                        <h3>Third Convolutional Layer</h3>
                        ${data.visualizations.conv3}
                    `;
                }
            }
            
            if (data.visualizations.summary) {
                const summaryPlot = document.getElementById('summary-plot');
                if (summaryPlot) {
                    summaryPlot.src = `data:image/png;base64,${data.visualizations.summary}`;
                }
            }
        }

        // Update layer info
        if (data.layer_info) {
            this.updateLayerInfo(data.layer_info);
        }

        // Scroll to visualizations
        if (vizSection) {
            vizSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    updateLayerInfo(layerInfo) {
        // Update conv1 info
        if (layerInfo.conv1) {
            const conv1FiltersEl = document.getElementById('conv1-filters');
            if (conv1FiltersEl) {
                conv1FiltersEl.textContent = layerInfo.conv1.num_filters || '32';
            }
            if (layerInfo.conv1.shape) {
                const conv1ShapeEl = document.getElementById('conv1-shape');
                if (conv1ShapeEl) {
                    conv1ShapeEl.textContent = `${layerInfo.conv1.shape[0]}×${layerInfo.conv1.shape[1]}`;
                }
            }
        }

        // Update conv2 info
        if (layerInfo.conv2) {
            const conv2FiltersEl = document.getElementById('conv2-filters');
            if (conv2FiltersEl) {
                conv2FiltersEl.textContent = layerInfo.conv2.num_filters || '64';
            }
            if (layerInfo.conv2.shape) {
                const conv2ShapeEl = document.getElementById('conv2-shape');
                if (conv2ShapeEl) {
                    conv2ShapeEl.textContent = `${layerInfo.conv2.shape[0]}×${layerInfo.conv2.shape[1]}`;
                }
            }
        }

        // Update conv3 info
        if (layerInfo.conv3) {
            const conv3FiltersEl = document.getElementById('conv3-filters');
            if (conv3FiltersEl) {
                conv3FiltersEl.textContent = layerInfo.conv3.num_filters || '128';
            }
            if (layerInfo.conv3.shape) {
                const conv3ShapeEl = document.getElementById('conv3-shape');
                if (conv3ShapeEl) {
                    conv3ShapeEl.textContent = `${layerInfo.conv3.shape[0]}×${layerInfo.conv3.shape[1]}`;
                }
            }
        }
    }

    showError(message) {
        // Hide visualization section on error
        const vizSection = document.getElementById('visualization-section');
        const probSection = document.getElementById('probability-section');
        if (vizSection) {
            vizSection.classList.add('hidden');
        }
        if (probSection) {
            probSection.classList.add('hidden');
        }
    }

    updateUncertaintyVisualizations(data) {
        // Show visualization section
        const vizSection = document.getElementById('visualization-section');
        if (vizSection) {
            vizSection.classList.remove('hidden');
        }

        // Show uncertainty tab
        const uncertaintyTabBtn = document.querySelector('.uncertainty-tab');
        if (uncertaintyTabBtn) {
            uncertaintyTabBtn.classList.remove('hidden');
        }

        // Update probability section with uncertainty info
        const probSection = document.getElementById('probability-section');
        if (probSection) {
            probSection.classList.remove('hidden');

            // Update chart with uncertainty visualization
            const probChart = document.getElementById('prob-chart');
            if (probChart && data.visualizations && data.visualizations.uncertainty_chart) {
                probChart.src = `data:image/png;base64,${data.visualizations.uncertainty_chart}`;
            }

            // Update explanation for uncertainty
            const explanation = probSection.querySelector('.explanation');
            if (explanation && data.uncertainty) {
                const level = this.getUncertaintyLevel(data.uncertainty.predictive_entropy);
                const ci = data.uncertainty.confidence_interval_95;
                explanation.innerHTML = `
                    <strong>Uncertainty Analysis:</strong>
                    Prediction confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong>
                    (95% CI: ${(ci.lower * 100).toFixed(1)}% - ${(ci.upper * 100).toFixed(1)}%)
                    <br>
                    Uncertainty level: <span class="uncertainty-${level}">${level.toUpperCase()}</span>
                    | Entropy: ${data.uncertainty.predictive_entropy.toFixed(4)}
                `;
            }
        }

        // Update uncertainty tab content
        const uncertaintyContent = document.getElementById('uncertainty-content');
        if (uncertaintyContent && data.visualizations) {
            let html = '';

            // Add summary HTML if available
            if (data.visualizations.summary_html) {
                html += data.visualizations.summary_html;
            }

            // Add uncertainty gauge
            if (data.visualizations.uncertainty_gauge) {
                html += `
                    <div class="uncertainty-gauge-section">
                        <h4>Uncertainty Gauges</h4>
                        <img src="data:image/png;base64,${data.visualizations.uncertainty_gauge}"
                             class="visualization-img" alt="Uncertainty gauges" />
                    </div>
                `;
            }

            // Add sample distribution
            if (data.visualizations.samples_distribution) {
                html += `
                    <div class="samples-distribution-section">
                        <h4>Monte Carlo Sample Distributions</h4>
                        <p class="explanation">
                            Each histogram shows the distribution of predicted probabilities
                            across ${data.n_samples || 50} forward passes with dropout enabled.
                        </p>
                        <img src="data:image/png;base64,${data.visualizations.samples_distribution}"
                             class="visualization-img" alt="Sample distributions" />
                    </div>
                `;
            }

            uncertaintyContent.innerHTML = html;
        }

        // Switch to uncertainty tab
        this.switchTab('uncertainty');

        // Scroll to visualizations
        if (vizSection) {
            vizSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    getUncertaintyLevel(entropy) {
        if (entropy < 0.3) return 'low';
        if (entropy < 1.0) return 'medium';
        return 'high';
    }
}

// Initialize visualizer
const visualizer = new NeuralNetworkVisualizer();

// Update the main.js predict function to use visualizer
// This would be integrated with the existing main.js code
window.updateVisualizationsCallback = (data, useUncertainty = false) => {
    if (useUncertainty) {
        visualizer.updateUncertaintyVisualizations(data);
    } else {
        visualizer.updateVisualizations(data);
    }
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
