"""
Activation visualization utilities for understanding neural network behavior.
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np


def create_probability_chart(predictions: list[float]) -> str:
    """
    Create a bar chart of prediction probabilities.

    Args:
        predictions: List of 10 probability values

    Returns:
        Base64 encoded chart image
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    digits = list(range(10))
    bars = ax.bar(
        digits, predictions, color="steelblue", edgecolor="black", linewidth=1
    )

    # Highlight the predicted class
    predicted_class = np.argmax(predictions)
    bars[predicted_class].set_color("darkgreen")

    # Add value labels on bars
    for i, (digit, prob) in enumerate(zip(digits, predictions, strict=False)):
        ax.text(
            digit, prob + 0.01, f"{prob:.1%}", ha="center", va="bottom", fontsize=10
        )

    # Styling
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Prediction Probabilities", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(digits)
    ax.grid(axis="y", alpha=0.3)

    # Add confidence line
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_activation_heatmap(
    original_image: np.ndarray, activation_map: np.ndarray, alpha: float = 0.5
) -> str:
    """
    Create a heatmap overlay on the original image.

    Args:
        original_image: Original input image (28, 28)
        activation_map: Activation map to overlay
        alpha: Transparency of overlay

    Returns:
        Base64 encoded image
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    ax1.imshow(original_image, cmap="gray")
    ax1.set_title("Original Input")
    ax1.axis("off")

    # Activation map
    # Resize activation map to match original image size
    from scipy.ndimage import zoom

    zoom_factor = (
        original_image.shape[0] / activation_map.shape[0],
        original_image.shape[1] / activation_map.shape[1],
    )
    activation_resized = zoom(activation_map, zoom_factor, order=1)

    ax2.imshow(activation_resized, cmap="hot")
    ax2.set_title("Activation Heatmap")
    ax2.axis("off")

    # Overlay
    ax3.imshow(original_image, cmap="gray")
    ax3.imshow(activation_resized, cmap="hot", alpha=alpha)
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_filter_grid_image(
    filters: np.ndarray,
    indices: list[int],
    mean_acts: np.ndarray,
    cmap: str = "viridis",
) -> str:
    """
    Create a 4x4 grid of filter activations as a standalone image.

    Args:
        filters: Filter activation data (height, width, num_filters)
        indices: Filter indices sorted by activation strength
        mean_acts: Mean activation values for each filter
        cmap: Colormap to use

    Returns:
        Base64 encoded image string
    """
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.patch.set_facecolor("white")

    max_activation = max(mean_acts) if len(mean_acts) > 0 else 1

    for i in range(16):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        if i < filters.shape[-1]:
            filter_activation = filters[:, :, i]
            filter_idx = indices[i]
            activation_strength = mean_acts[filter_idx]

            # Normalize for display
            vmin, vmax = filter_activation.min(), filter_activation.max()
            if vmax > vmin:
                filter_normalized = (filter_activation - vmin) / (vmax - vmin)
            else:
                filter_normalized = filter_activation

            # Plot with colormap
            im = ax.imshow(filter_normalized, cmap=cmap, vmin=0, vmax=1)

            # Add border based on activation strength
            strength_percentile = (
                activation_strength / max_activation if max_activation > 0 else 0
            )
            if strength_percentile > 0.8:
                border_color, border_width = "red", 3
            elif strength_percentile > 0.5:
                border_color, border_width = "orange", 2
            else:
                border_color, border_width = "gray", 1

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            ax.set_title(f"F{filter_idx}", fontsize=8, pad=3)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=0.5)

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_activation_legend_image(cmap: str = "viridis") -> str:
    """
    Create a standalone colorbar legend image.

    Args:
        cmap: Colormap to create legend for

    Returns:
        Base64 encoded image string
    """
    fig, ax = plt.subplots(figsize=(1.5, 6))
    fig.patch.set_facecolor("white")

    # Create colorbar
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 256)

    # Add labels
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_yticklabels(["Very Low", "Low", "Medium", "High", "Very High"], fontsize=11)
    ax.set_xticks([])
    ax.set_ylabel("Activation Strength", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_activation_html(activations: dict[str, any], layer_name: str) -> str:
    """
    Create an HTML visualization of layer activations with embedded images.

    Args:
        activations: Processed activations dictionary
        layer_name: Name of layer to visualize

    Returns:
        HTML string with embedded base64 images
    """
    # Get prediction for context
    predicted_digit = activations["predictions"]["predicted_class"]
    confidence = activations["predictions"]["confidence"]

    # Layer-specific configurations
    layer_info = {
        "conv1": {
            "title": "First Convolutional Layer (32 filters)",
            "description": "Filter activations from the first convolutional layer processing the input image",
            "cmap": "RdBu_r",
            "details": "32 filters with 3×3 kernels processing the input image",
        },
        "conv2": {
            "title": "Second Convolutional Layer (64 filters)",
            "description": "Filter activations from the second convolutional layer",
            "cmap": "viridis",
            "details": "64 filters processing outputs from the first layer",
        },
        "conv3": {
            "title": "Third Convolutional Layer (128 filters)",
            "description": "Filter activations from the third convolutional layer",
            "cmap": "plasma",
            "details": "128 filters in the final convolutional stage",
        },
    }

    info = layer_info.get(
        layer_name,
        {
            "title": f"{layer_name} Activations",
            "description": "Neural network layer activations",
            "cmap": "viridis",
            "details": "Neural network layer processing",
        },
    )

    # Generate component images if layer data exists
    if layer_name in activations and "top_filters" in activations[layer_name]:
        filters = activations[layer_name]["top_filters"]
        indices = activations[layer_name]["top_indices"]
        mean_acts = activations[layer_name]["mean_activations"]

        # Create individual images
        filter_grid_img = create_filter_grid_image(
            filters, indices, mean_acts, info["cmap"]
        )
        legend_img = create_activation_legend_image(info["cmap"])

        # Calculate statistics
        num_active_filters = sum(1 for act in mean_acts if act > np.mean(mean_acts))
        max_activation = max(mean_acts) if len(mean_acts) > 0 else 0
        avg_activation = np.mean(mean_acts) if len(mean_acts) > 0 else 0

    else:
        filter_grid_img = legend_img = ""
        num_active_filters = max_activation = avg_activation = 0

    # Generate HTML template
    html_template = f"""
    <div class="activation-container" data-layer="{layer_name}">
        <div class="activation-header">
            <h2 class="layer-title">{info["title"]}</h2>
            <div class="prediction-context">
                <span class="predicted-digit">Predicted: <strong>{predicted_digit}</strong></span>
                <span class="confidence">Confidence: <strong>{confidence:.1%}</strong></span>
            </div>
        </div>
        
        <div class="layer-description">
            <p>{info["description"]}</p>
            <div class="layer-details">{info["details"]}</div>
        </div>
        
        <div class="activation-content">
            <div class="filter-section">
                <h3>Filter Activations</h3>
                <div class="filter-visualization">
                    <div class="filter-grid-container">
                        <img src="data:image/png;base64,{filter_grid_img}" 
                             alt="Filter activation grid" 
                             class="filter-grid-image" />
                    </div>
                    <div class="legend-container">
                        <img src="data:image/png;base64,{legend_img}" 
                             alt="Activation strength legend" 
                             class="legend-image" />
                    </div>
                </div>
                
                <div class="filter-stats">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Active Filters:</span>
                            <span class="stat-value">{num_active_filters}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Max Activation:</span>
                            <span class="stat-value">{max_activation:.3f}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Avg Activation:</span>
                            <span class="stat-value">{avg_activation:.3f}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="interpretation-guide">
            <h4>Reading the visualization:</h4>
            <ul class="guide-list">
                <li><strong>Brighter areas</strong> = stronger activation (the network "sees" something important)</li>
                <li><strong>Red borders</strong> = filters most active for this input (key features for recognition)</li>
                <li><strong>Each filter</strong> has learned to detect a specific type of feature</li>
                <li><strong>{info["details"]}</strong></li>
            </ul>
        </div>
    </div>
    """

    return html_template.strip()


def create_layer_summary_plot(activations: dict[str, any]) -> str:
    """
    Create a summary visualization showing how activations flow to the prediction.

    Args:
        activations: Processed activations from all layers

    Returns:
        Base64 encoded image
    """
    # Get prediction info
    predicted_digit = activations["predictions"]["predicted_class"]
    confidence = activations["predictions"]["confidence"]

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots with better proportions
    gs = fig.add_gridspec(
        3, 4, hspace=0.5, wspace=0.4, left=0.08, right=0.95, top=0.9, bottom=0.15
    )

    # Title with prediction
    fig.suptitle(
        f'How the Network Recognized "{predicted_digit}" (Confidence: {confidence:.1%})',
        fontsize=18,
        fontweight="bold",
    )

    # 1. Input Processing Flow (top row)
    ax1 = fig.add_subplot(gs[0, :])

    # Show processing stages
    stages = [
        "Input\n(28×28)",
        "Layer 1\n(Edge Detection)",
        "Layer 2\n(Shape Detection)",
        "Layer 3\n(Digit Features)",
        "Dense Layer\n(Classification)",
        f'Output\n"{predicted_digit}"',
    ]
    stage_colors = [
        "gray",
        "lightblue",
        "skyblue",
        "steelblue",
        "lightcoral",
        "darkgreen",
    ]

    y_pos = 0.5
    for i, (stage, color) in enumerate(zip(stages, stage_colors, strict=False)):
        x_pos = i / (len(stages) - 1)

        # Draw box
        box = plt.Rectangle(
            (x_pos - 0.07, y_pos - 0.25),
            0.14,
            0.5,
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ax1.add_patch(box)

        # Add text
        ax1.text(
            x_pos,
            y_pos,
            stage,
            ha="center",
            va="center",
            fontweight="bold" if i == len(stages) - 1 else "normal",
            fontsize=11,
        )

        # Draw arrow
        if i < len(stages) - 1:
            ax1.arrow(
                x_pos + 0.07,
                y_pos,
                0.16,
                0,
                head_width=0.08,
                head_length=0.04,
                fc="black",
                ec="black",
            )

    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Information Flow Through the Network", fontsize=14, pad=20)

    # 2. Layer Activations (middle row)
    # Conv layers
    conv_layers = ["conv1", "conv2", "conv3"]
    for i, layer_name in enumerate(conv_layers):
        ax = fig.add_subplot(gs[1, i])

        if layer_name in activations:
            layer_data = activations[layer_name]
            mean_acts = np.array(layer_data.get("mean_activations", []))

            if len(mean_acts) > 0:
                # Show only top activated filters
                top_n = 10
                top_indices = np.argsort(mean_acts)[-top_n:][::-1]
                top_values = mean_acts[top_indices]

                # Color bars based on activation strength
                colors = plt.cm.viridis(top_values / top_values.max())

                bars = ax.bar(range(len(top_values)), top_values, color=colors)
                ax.set_title(
                    f"{layer_name.upper()}: Top {top_n} Active Filters\n(of {layer_data['num_filters']} total)",
                    fontsize=12,
                    pad=15,
                )
                ax.set_xlabel("Most Active Filters", fontsize=11)
                ax.set_ylabel("Activation Strength", fontsize=11)
                ax.set_ylim(bottom=0)

                # Use simple filter numbers instead of fake feature names
                filter_labels = [f"F{top_indices[i]}" for i in range(len(top_values))]

                ax.set_xticks(range(len(top_values)))
                ax.set_xticklabels(filter_labels, rotation=45, ha="right", fontsize=10)

    # Dense layer - show connection to digits
    ax_dense = fig.add_subplot(gs[1, 3])
    if "dense" in activations:
        dense_acts = np.array(activations["dense"]["activations"])

        # Group neurons by their contribution to each digit
        # This is simplified - in reality we'd need the weight matrix
        top_n = 10
        top_indices = np.argsort(dense_acts)[-top_n:][::-1]
        top_values = dense_acts[top_indices]

        colors = plt.cm.Reds(top_values / top_values.max())
        bars = ax_dense.bar(range(len(top_values)), top_values, color=colors)
        ax_dense.set_title(
            f"Dense Layer: Top {top_n} Active Neurons\n(of {len(dense_acts)} total)",
            fontsize=12,
            pad=15,
        )
        ax_dense.set_xlabel("Decision Neurons", fontsize=11)
        ax_dense.set_ylabel("Activation", fontsize=11)

        # Use simple neuron numbers instead of fake labels
        neuron_labels = [f"N{top_indices[i]}" for i in range(len(top_values))]

        ax_dense.set_xticks(range(len(top_values)))
        ax_dense.set_xticklabels(neuron_labels, rotation=45, ha="right", fontsize=10)

    # 3. Final Decision (bottom row - centered)
    ax_decision = fig.add_subplot(gs[2, 1:3])

    # Show how activations lead to final probabilities
    probs = activations["predictions"]["probabilities"]
    digits = list(range(10))

    # Color code by probability
    colors = [
        "darkgreen"
        if i == predicted_digit
        else "orange"
        if probs[i] > 0.1
        else "lightgray"
        for i in digits
    ]

    bars = ax_decision.bar(digits, probs, color=colors, edgecolor="black", linewidth=1)

    # Add percentage labels
    for i, (digit, prob) in enumerate(zip(digits, probs, strict=False)):
        if prob > 0.01:  # Only show labels for non-negligible probabilities
            ax_decision.text(
                digit,
                prob + 0.01,
                f"{prob:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if digit == predicted_digit else "normal",
            )

    ax_decision.set_xlabel("Digit", fontsize=12)
    ax_decision.set_ylabel("Probability", fontsize=12)
    ax_decision.set_title("Final Classification Probabilities", fontsize=14)
    ax_decision.set_ylim(0, 1.1)
    ax_decision.set_xticks(digits)
    ax_decision.grid(axis="y", alpha=0.3)

    # Add legend outside the plot area to avoid any overlap
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="darkgreen", label="Predicted"),
        Patch(facecolor="orange", label="Alternative (>10%)"),
        Patch(facecolor="lightgray", label="Unlikely (<10%)"),
    ]
    ax_decision.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_uncertainty_chart(
    mean_probs: list[float],
    variances: list[float],
    confidence_interval: tuple[float, float],
    predicted_class: int,
) -> str:
    """
    Create a bar chart showing predictions with uncertainty bands.

    Args:
        mean_probs: Mean probability for each class
        variances: Variance for each class from MC samples
        confidence_interval: 95% CI for predicted class
        predicted_class: The predicted digit

    Returns:
        Base64 encoded chart image
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    digits = list(range(10))
    std_devs = [np.sqrt(v) for v in variances]

    # Create bars with error bars for uncertainty
    bars = ax.bar(
        digits,
        mean_probs,
        yerr=std_devs,
        capsize=5,
        color="steelblue",
        edgecolor="black",
        linewidth=1,
        ecolor="darkred",
        alpha=0.8,
    )

    # Highlight the predicted class
    bars[predicted_class].set_color("darkgreen")

    # Add value labels on bars
    for i, (prob, std) in enumerate(zip(mean_probs, std_devs, strict=False)):
        label = f"{prob:.1%}\n±{std:.1%}"
        ax.text(i, prob + std + 0.02, label, ha="center", va="bottom", fontsize=9)

    # Styling
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"MC Dropout Predictions with Uncertainty\n"
        f"95% CI for digit {predicted_class}: "
        f"[{confidence_interval[0]:.1%}, {confidence_interval[1]:.1%}]",
        fontsize=14,
    )
    ax.set_ylim(0, min(1.3, max(mean_probs) + max(std_devs) + 0.15))
    ax.set_xticks(digits)
    ax.grid(axis="y", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="darkgreen", label=f"Predicted: {predicted_class}"),
        Patch(facecolor="steelblue", label="Other digits"),
        Line2D([0], [0], color="darkred", label="Uncertainty (±1 std)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_mc_samples_distribution(
    all_predictions: np.ndarray, predicted_class: int
) -> str:
    """
    Create a visualization showing the distribution of MC samples.

    Args:
        all_predictions: Array of shape (n_samples, 10) with all predictions
        predicted_class: The predicted digit

    Returns:
        Base64 encoded chart image
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(
        "MC Dropout Sample Distributions by Digit", fontsize=14, fontweight="bold"
    )

    for digit in range(10):
        ax = axes[digit // 5, digit % 5]
        samples = all_predictions[:, digit]

        # Create histogram
        color = "darkgreen" if digit == predicted_class else "steelblue"
        ax.hist(samples, bins=20, color=color, edgecolor="black", alpha=0.7)

        # Add mean line
        mean_val = np.mean(samples)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label="Mean")

        # Add 95% CI lines
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        ax.axvline(ci_low, color="orange", linestyle=":", linewidth=1.5)
        ax.axvline(ci_high, color="orange", linestyle=":", linewidth=1.5)

        ax.set_title(
            f"Digit {digit}",
            fontweight="bold" if digit == predicted_class else "normal",
        )
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_uncertainty_gauge(
    predictive_entropy: float, mutual_information: float, max_entropy: float = 2.303
) -> str:
    """
    Create a gauge visualization for uncertainty metrics.

    Args:
        predictive_entropy: Total uncertainty (entropy)
        mutual_information: Model uncertainty
        max_entropy: Maximum possible entropy (log(10) for 10 classes)

    Returns:
        Base64 encoded chart image
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Normalize values to [0, 1]
    entropy_normalized = min(predictive_entropy / max_entropy, 1.0)
    mi_normalized = min(mutual_information / max_entropy, 1.0)

    metrics = [
        (
            "Predictive Entropy",
            entropy_normalized,
            predictive_entropy,
            "Total Uncertainty",
        ),
        ("Mutual Information", mi_normalized, mutual_information, "Model Uncertainty"),
    ]

    for ax, (name, normalized, raw_val, description) in zip(
        axes, metrics, strict=False
    ):
        # Create gauge background
        theta = np.linspace(0, np.pi, 100)
        r_outer = 1.0
        r_inner = 0.6

        # Background arc (gray)
        ax.fill_between(
            np.cos(theta) * r_outer,
            np.sin(theta) * r_outer,
            np.cos(theta) * r_inner,
            np.sin(theta) * r_inner,
            color="lightgray",
            alpha=0.5,
        )

        # Colored arc based on value
        theta_filled = np.linspace(np.pi, np.pi - normalized * np.pi, 50)

        # Color gradient from green (low) to red (high)
        if normalized < 0.3:
            color = "green"
            level = "Low"
        elif normalized < 0.6:
            color = "orange"
            level = "Medium"
        else:
            color = "red"
            level = "High"

        ax.fill_between(
            np.cos(theta_filled) * r_outer,
            np.sin(theta_filled) * r_outer,
            np.cos(theta_filled) * r_inner,
            np.sin(theta_filled) * r_inner,
            color=color,
            alpha=0.8,
        )

        # Add needle
        needle_angle = np.pi - normalized * np.pi
        ax.plot(
            [0, 0.5 * np.cos(needle_angle)],
            [0, 0.5 * np.sin(needle_angle)],
            color="black",
            linewidth=3,
        )
        ax.plot(0, 0, "ko", markersize=10)

        # Labels
        ax.text(0, -0.2, f"{raw_val:.4f}", ha="center", fontsize=14, fontweight="bold")
        ax.text(0, -0.4, f"({level})", ha="center", fontsize=12, color=color)
        ax.set_title(f"{name}\n{description}", fontsize=12, pad=10)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_mc_dropout_summary_html(mc_result: dict) -> str:
    """
    Create an HTML summary of MC Dropout results.

    Args:
        mc_result: Dictionary from MCDropoutResult.to_dict()

    Returns:
        HTML string with MC Dropout summary
    """
    predicted = mc_result["predicted_class"]
    confidence = mc_result["confidence"]
    entropy = mc_result["uncertainty"]["predictive_entropy"]
    mi = mc_result["uncertainty"]["mutual_information"]
    ci = mc_result["uncertainty"]["confidence_interval_95"]
    n_samples = mc_result["num_samples"]

    # Determine uncertainty level
    if entropy < 0.3:
        level = "low"
        level_color = "#28a745"
        level_desc = "The model is confident in this prediction."
    elif entropy < 1.0:
        level = "medium"
        level_color = "#ffc107"
        level_desc = "Moderate uncertainty. Consider reviewing the input."
    else:
        level = "high"
        level_color = "#dc3545"
        level_desc = "High uncertainty. This prediction may be unreliable."

    html = f"""
    <div class="mc-dropout-summary">
        <div class="mc-header">
            <h2>Monte Carlo Dropout Analysis</h2>
            <p class="mc-subtitle">Uncertainty estimation from {n_samples} forward passes</p>
        </div>

        <div class="mc-prediction">
            <div class="prediction-main">
                <span class="predicted-digit-large">{predicted}</span>
                <div class="prediction-details">
                    <div class="confidence-value">{confidence:.1%}</div>
                    <div class="confidence-label">Mean Confidence</div>
                </div>
            </div>
            <div class="confidence-interval">
                <span class="ci-label">95% Confidence Interval:</span>
                <span class="ci-value">[{ci["lower"]:.1%}, {ci["upper"]:.1%}]</span>
            </div>
        </div>

        <div class="uncertainty-metrics">
            <h3>Uncertainty Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{entropy:.4f}</div>
                    <div class="metric-label">Predictive Entropy</div>
                    <div class="metric-desc">Total uncertainty</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{mi:.4f}</div>
                    <div class="metric-label">Mutual Information</div>
                    <div class="metric-desc">Model uncertainty</div>
                </div>
                <div class="metric-card uncertainty-{level}">
                    <div class="metric-value" style="color: {level_color}">{level.upper()}</div>
                    <div class="metric-label">Uncertainty Level</div>
                    <div class="metric-desc">{level_desc}</div>
                </div>
            </div>
        </div>

        <div class="variance-section">
            <h3>Per-Class Variance</h3>
            <div class="variance-bars">
    """

    for digit, var in mc_result["variance_per_class"].items():
        prob = mc_result["mean_probabilities"][digit]
        bar_width = min(prob * 100, 100)
        is_predicted = int(digit) == predicted
        bar_class = "variance-bar predicted" if is_predicted else "variance-bar"

        html += f"""
                <div class="variance-row">
                    <span class="digit-label">{digit}</span>
                    <div class="bar-container">
                        <div class="{bar_class}" style="width: {bar_width}%"></div>
                    </div>
                    <span class="prob-value">{prob:.1%}</span>
                    <span class="var-value">±{np.sqrt(var):.3f}</span>
                </div>
        """

    html += """
            </div>
        </div>

        <div class="mc-interpretation">
            <h3>Interpretation Guide</h3>
            <ul>
                <li><strong>Predictive Entropy:</strong> Measures total uncertainty. Higher values indicate the model is less certain about the prediction.</li>
                <li><strong>Mutual Information:</strong> Captures model (epistemic) uncertainty - uncertainty that could be reduced with more training data.</li>
                <li><strong>Confidence Interval:</strong> The range in which the true probability likely falls (95% of the time).</li>
                <li><strong>Variance:</strong> How much each class probability varies across MC samples. Higher variance = less stable prediction.</li>
            </ul>
        </div>
    </div>
    """

    return html
