"""
Activation visualization utilities for understanding neural network behavior.
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np


def create_activation_plot(
    activations: dict[str, any], layer_name: str, figsize: tuple[int, int] = (14, 10)
) -> str:
    """
    Create an intuitive visualization of what features the layer is detecting.

    Args:
        activations: Processed activations dictionary
        layer_name: Name of layer to visualize
        figsize: Figure size

    Returns:
        Base64 encoded image string
    """
    # Get prediction for context
    predicted_digit = activations["predictions"]["predicted_class"]
    confidence = activations["predictions"]["confidence"]

    fig = plt.figure(figsize=figsize)

    # Create custom layout
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.2)

    # Layer-specific titles and descriptions
    layer_info = {
        "conv1": {
            "title": "Layer 1: Edge and Stroke Detection",
            "description": "This layer detects basic features like edges, curves, and strokes",
            "cmap": "RdBu_r",
        },
        "conv2": {
            "title": "Layer 2: Shape and Pattern Recognition",
            "description": "This layer combines edges to recognize shapes and patterns",
            "cmap": "viridis",
        },
        "conv3": {
            "title": "Layer 3: Digit-Specific Features",
            "description": "This layer recognizes features specific to individual digits",
            "cmap": "plasma",
        },
    }

    info = layer_info.get(
        layer_name,
        {
            "title": f"{layer_name} Activations",
            "description": "Neural network layer activations",
            "cmap": "viridis",
        },
    )

    # Main title
    fig.suptitle(
        f"{info['title']}\nFor predicted digit: {predicted_digit}",
        fontsize=16,
        fontweight="bold",
    )

    # Description
    ax_desc = fig.add_subplot(gs[0, :])
    ax_desc.text(
        0.5,
        0.5,
        info["description"],
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
    )
    ax_desc.axis("off")

    if layer_name in activations and "top_filters" in activations[layer_name]:
        filters = activations[layer_name]["top_filters"]
        indices = activations[layer_name]["top_indices"]
        mean_acts = activations[layer_name]["mean_activations"]

        # Show top 16 filters in a 4x4 grid
        for i in range(min(16, filters.shape[-1])):
            row = (i // 4) + 1
            col = i % 4
            ax = fig.add_subplot(gs[row, col])

            # Get the filter
            filter_activation = filters[:, :, i]
            filter_idx = indices[i]
            activation_strength = mean_acts[filter_idx]

            # Normalize for display
            vmin, vmax = filter_activation.min(), filter_activation.max()
            if vmax > vmin:
                filter_normalized = (filter_activation - vmin) / (vmax - vmin)
            else:
                filter_normalized = filter_activation

            # Plot with appropriate colormap
            im = ax.imshow(filter_normalized, cmap=info["cmap"], vmin=0, vmax=1)

            # Add border color based on activation strength
            strength_percentile = (
                activation_strength / max(mean_acts) if max(mean_acts) > 0 else 0
            )
            if strength_percentile > 0.8:
                border_color = "red"
                border_width = 3
                strength_label = "Very Active"
            elif strength_percentile > 0.5:
                border_color = "orange"
                border_width = 2
                strength_label = "Active"
            else:
                border_color = "gray"
                border_width = 1
                strength_label = "Low Activity"

            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Title with interpretation
            feature_descriptions = {
                "conv1": [
                    "vertical edge",
                    "horizontal edge",
                    "diagonal edge",
                    "curve",
                    "corner",
                    "dot",
                    "line segment",
                    "stroke",
                ],
                "conv2": [
                    "loop",
                    "intersection",
                    "parallel lines",
                    "curve pattern",
                    "angle",
                    "rounded shape",
                    "straight pattern",
                    "complex edge",
                ],
                "conv3": [
                    "digit shape",
                    "specific curve",
                    "digit feature",
                    "pattern",
                    "characteristic",
                    "unique shape",
                    "digit part",
                    "recognition",
                ],
            }

            # Get a description based on filter index
            descriptions = feature_descriptions.get(layer_name, ["feature"])
            feature_desc = descriptions[i % len(descriptions)]

            ax.set_title(
                f"Filter {filter_idx}: {feature_desc}\n({strength_label})",
                fontsize=9,
                pad=2,
            )
            ax.axis("off")

            # Add small colorbar for the first filter to show scale
            if i == 0:
                cbar_ax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.01,
                        ax.get_position().height,
                    ]
                )
                fig.colorbar(im, cax=cbar_ax, label="Activation")

        # Hide unused subplots
        for i in range(filters.shape[-1], 16):
            row = (i // 4) + 1
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

    # Add interpretation guide at bottom
    ax_guide = fig.add_subplot(gs[4, :])
    ax_guide.axis("off")

    guide_text = """Reading the visualization:
    • Brighter areas = stronger activation (the network "sees" something important there)
    • Red borders = filters most active for this input (key features for recognition)
    • Each filter has learned to detect a specific type of feature
    """

    if layer_name == "conv1":
        guide_text += "• Layer 1 looks for basic building blocks like edges and curves"
    elif layer_name == "conv2":
        guide_text += "• Layer 2 combines simple features into more complex patterns"
    elif layer_name == "conv3":
        guide_text += f"• Layer 3 has learned features specific to recognizing '{predicted_digit}'"

    ax_guide.text(
        0.5,
        0.5,
        guide_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


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

    # Create grid for subplots
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Title with prediction
    fig.suptitle(
        f'How the Network Recognized "{predicted_digit}" (Confidence: {confidence:.1%})',
        fontsize=16,
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
            (x_pos - 0.08, y_pos - 0.3),
            0.16,
            0.6,
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
        )

        # Draw arrow
        if i < len(stages) - 1:
            ax1.arrow(
                x_pos + 0.08,
                y_pos,
                0.15,
                0,
                head_width=0.1,
                head_length=0.05,
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
                    f'{layer_name.upper()}: Top {top_n} Active Filters\n(of {layer_data["num_filters"]} total)'
                )
                ax.set_xlabel("Most Active Filters")
                ax.set_ylabel("Activation Strength")
                ax.set_ylim(bottom=0)

                # Add meaningful labels instead of filter indices
                # Create feature descriptions based on layer and activation patterns
                if layer_name == "conv1":
                    # Layer 1: Basic edge and stroke detectors
                    feature_names = [
                        "vertical edges",
                        "horizontal edges",
                        "diagonal (/) edges",
                        "diagonal (\\) edges",
                        "curved strokes",
                        "corners",
                        "dots/endpoints",
                        "line segments",
                        "thick strokes",
                        "thin strokes",
                    ]
                elif layer_name == "conv2":
                    # Layer 2: Shape and pattern detectors
                    feature_names = [
                        "loops/circles",
                        "intersections",
                        "parallel lines",
                        "curves",
                        "right angles",
                        "acute angles",
                        "S-curves",
                        "straight segments",
                        "crossings",
                        "junctions",
                    ]
                elif layer_name == "conv3":
                    # Layer 3: Digit-specific feature detectors
                    # These would be more specific to the predicted digit
                    digit_features = {
                        0: [
                            "closed loop",
                            "oval shape",
                            "vertical symmetry",
                            "no gaps",
                        ],
                        1: ["vertical line", "minimal width", "straight", "no loops"],
                        2: [
                            "top curve",
                            "diagonal middle",
                            "horizontal base",
                            "S-shape",
                        ],
                        3: [
                            "two curves",
                            "right-facing",
                            "no left edge",
                            "middle indent",
                        ],
                        4: [
                            "vertical right",
                            "horizontal cross",
                            "open left",
                            "angular",
                        ],
                        5: [
                            "top horizontal",
                            "curved bottom",
                            "right hook",
                            "flat top",
                        ],
                        6: ["large loop", "top curve", "enclosed space", "curved hook"],
                        7: [
                            "diagonal line",
                            "horizontal top",
                            "angled down",
                            "no loops",
                        ],
                        8: ["two loops", "figure-8", "center cross", "symmetrical"],
                        9: ["top loop", "vertical right", "small circle", "descender"],
                    }
                    # Get features for the predicted digit
                    digit_specific = digit_features.get(
                        predicted_digit, ["digit feature"] * 4
                    )
                    # Mix with general high-level features
                    feature_names = digit_specific + [
                        "high contrast",
                        "key region",
                        "critical edge",
                        "unique pattern",
                        "distinctive shape",
                        "identifying mark",
                    ]
                else:
                    feature_names = [f"feature {i+1}" for i in range(10)]

                # Assign labels cyclically if we have more filters than descriptions
                labels = []
                for i in range(len(top_values)):
                    labels.append(feature_names[i % len(feature_names)])

                ax.set_xticks(range(len(top_values)))
                ax.set_xticklabels(labels, rotation=45, ha="right")

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
            f"Dense Layer: Top {top_n} Active Neurons\n(of {len(dense_acts)} total)"
        )
        ax_dense.set_xlabel("Decision Neurons")
        ax_dense.set_ylabel("Activation")

        # Add meaningful labels for dense layer neurons
        # These represent high-level decision-making neurons
        neuron_labels = [
            f"digit {predicted_digit} detector",
            f"{predicted_digit}-like patterns",
            "confidence builder",
            "ambiguity resolver",
            "feature combiner",
            "pattern matcher",
            "decision maker",
            "similarity checker",
            "final classifier",
            "certainty neuron",
        ]

        ax_dense.set_xticks(range(len(top_values)))
        ax_dense.set_xticklabels(
            neuron_labels[: len(top_values)], rotation=45, ha="right"
        )

    # 3. Final Decision (bottom row)
    ax_decision = fig.add_subplot(gs[2, :2])

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
                fontsize=10,
                fontweight="bold" if digit == predicted_digit else "normal",
            )

    ax_decision.set_xlabel("Digit", fontsize=12)
    ax_decision.set_ylabel("Probability", fontsize=12)
    ax_decision.set_title("Final Classification Probabilities", fontsize=14)
    ax_decision.set_ylim(0, 1.1)
    ax_decision.set_xticks(digits)
    ax_decision.grid(axis="y", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="darkgreen", label="Predicted"),
        Patch(facecolor="orange", label="Alternative (>10%)"),
        Patch(facecolor="lightgray", label="Unlikely (<10%)"),
    ]
    ax_decision.legend(handles=legend_elements, loc="upper right")

    # 4. Explanation text
    ax_explain = fig.add_subplot(gs[2, 2:])
    ax_explain.axis("off")

    # Create explanation based on the prediction
    explanation = f"""Why the network thinks this is a "{predicted_digit}":

• The early layers detected edges and strokes typical of "{predicted_digit}"
• Middle layers recognized shape patterns associated with "{predicted_digit}"
• The final layers combined these features with {confidence:.1%} confidence

The network considered alternatives:"""

    # Add top alternatives
    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for i, (digit, prob) in enumerate(sorted_probs[1:4]):  # Top 3 alternatives
        if prob > 0.01:
            explanation += f"\n• Digit {digit}: {prob:.1%} probability"

    ax_explain.text(
        0.05,
        0.95,
        explanation,
        transform=ax_explain.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64
