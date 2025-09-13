# Learning Machine Learning with MNIST 🧠

**An educational journey into deep learning through handwritten digit recognition**

Welcome! This project is designed to help you understand machine learning concepts by building a complete digit classifier from scratch. Whether you're curious about how neural networks work or want hands-on experience with ML, this project walks you through every step.

## 🎓 What You'll Learn

By exploring this project, you'll understand:
- **How neural networks "see" and process images**
- **What happens inside a convolutional neural network (CNN)**
- **How data flows through an ML pipeline**
- **Why certain architectural choices matter**
- **How to visualize what your model is actually learning**

## 🧩 The Learning Journey

This project is structured as a step-by-step exploration:

1. **Start Simple**: Load and explore the famous MNIST dataset
2. **Build Intuition**: Understand how images become numbers
3. **Create Intelligence**: Design a neural network architecture
4. **Watch It Learn**: Train your model and see it improve
5. **Look Inside**: Visualize what each layer actually learns
6. **Make It Interactive**: Build a web interface to test your understanding

## 🔍 What Makes This Educational?

- **Complete transparency**: Every step is explained and visible
- **Visual learning**: See your neural network's "thoughts" in real-time
- **Hands-on experimentation**: Modify parameters and see immediate results
- **No magic**: Understanding why each component exists and how it works
- **Progressive complexity**: Start simple, build up to sophisticated concepts

## ⚡ Quick Start (5 minutes to see magic happen!)

**Prerequisites**: Python 3.13+, basic comfort with command line

1. **Get the code**:
   ```bash
   git clone <repository-url>
   cd mnist_classifier_clean
   ```

2. **Set up your environment** (one-time setup):
   ```bash
   # Create isolated Python environment
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   
   # Install everything you need
   uv sync --all-extras
   ```

3. **See the data** (understand what we're working with):
   ```bash
   python quick_setup_data.py
   # This downloads MNIST and shows you sample digits
   ```

4. **Train your first neural network**:
   ```bash
   # Quick training to see it work (5 epochs, ~2 minutes)
   ./train_gpu.sh --quick
   
   # Watch the accuracy improve each epoch!
   ```

5. **Explore what your network learned**:
   ```bash
   # Start the interactive web interface
   python run_web_app.py
   
   # Open http://localhost:8000 and draw digits!
   # See what each layer of your network is thinking
   ```

That's it! You now have a working neural network that can recognize handwritten digits, and you can see exactly how it makes decisions.

## 🗺️ Exploring the Codebase (Your Learning Map)

The code is organized to follow your learning journey:

```
mnist_classifier_clean/
├── 📊 Data Pipeline
│   ├── quick_setup_data.py          # Start here! See the MNIST dataset
│   └── src/mnist_classifier/data/   # How we load and prepare images
│
├── 🧠 The Neural Network
│   ├── src/mnist_classifier/models/ # CNN architecture (the "brain")
│   └── src/mnist_classifier/training/ # How the network learns
│
├── 🎯 Making Predictions
│   ├── predict_digit.py             # Test single images
│   └── src/mnist_classifier/cli/    # Command-line tools
│
├── 🌐 Interactive Experience
│   ├── run_web_app.py              # Start the web interface
│   ├── templates/                   # Web pages you'll see
│   └── static/                      # Styles and JavaScript
│
└── 🔍 Neural Network Visualization
    └── src/mnist_classifier/visualization/ # See inside the "brain"
```

**Learning tip**: Start with `quick_setup_data.py`, then look at the model architecture in `src/mnist_classifier/models/cnn_model.py` to understand what you're building!

## 🎮 Fun Things to Try Once You're Running

### Experiment with Your Model

```bash
# Test your model on individual images
python predict_digit.py your_drawing.png

# Create your own test images from MNIST
python create_test_images.py my_test_images/ --mnist

# Process many images at once
python batch_predict.py test_images/*.png
```

### Watch Your Network Think

1. **Draw a digit** in the web interface
2. **See the predictions** change in real-time
3. **Explore each layer** - click the tabs to see what different parts of the network focus on
4. **Try tricky digits** - see where your network gets confused!

### Deep Dive: Understanding the Architecture

The neural network has three main parts:
- **Conv1**: Detects basic edges and shapes (32 filters)
- **Conv2**: Combines edges into more complex patterns (64 filters)  
- **Conv3**: Recognizes digit-specific features (64 filters)
- **Dense**: Makes the final decision (10 outputs, one per digit)

**Question to explore**: Why do we use progressively more filters in deeper layers?

## 🚀 Your Learning Journey (All Complete!)

This project was built step-by-step to teach ML concepts:

- ✅ **Foundation**: Set up Python environment and tools
- ✅ **Data Understanding**: Explore the MNIST dataset  
- ✅ **Neural Network Design**: Build a CNN architecture
- ✅ **Training**: Watch your model learn and improve
- ✅ **Command Line Tools**: Test your model programmatically
- ✅ **Web Interface**: Interactive digit drawing and prediction
- ✅ **Visualization**: See inside your neural network's "mind"

## 🧠 Key ML Concepts You'll Understand

By working through this project, you'll grasp:

- **Convolutional layers**: Why they're perfect for images
- **Activation functions**: How neurons decide what to "fire"
- **Backpropagation**: How networks learn from mistakes
- **Overfitting**: Why more training isn't always better
- **Feature visualization**: What patterns your network detects
- **Transfer learning principles**: How networks build hierarchical understanding

## 🤔 Questions to Explore

- Why does the network get confused between 4 and 9?
- What happens if you train for too many epochs?
- How would you modify this for letters instead of digits?
- Can you spot what Conv1 vs Conv3 focuses on?

## 📚 Want to Learn More?

- **[3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)**: Beautiful visual explanations
- **[Fast.ai Course](https://course.fast.ai/)**: Practical deep learning
- **[TensorFlow Tutorials](https://www.tensorflow.org/tutorials)**: Official documentation
- **[MNIST Database](http://yann.lecun.com/exdb/mnist/)**: The original dataset description

## 🎯 This is Educational!

Feel free to:
- Modify the architecture and see what happens
- Experiment with different learning rates
- Add your own visualizations
- Break things and fix them - that's how you learn!

**Remember**: The goal isn't just working code, it's understanding *why* it works.
