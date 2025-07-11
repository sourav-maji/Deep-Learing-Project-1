# Deep Learning Project 1 - Single Hidden Layer Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sourav-maji/Deep-Learing-Project-1/blob/main/Untitled1.ipynb)

## Description

This project demonstrates the fundamental concepts of deep learning by implementing a minimal neural network architecture for sentiment analysis. Using the famous IMDB movie reviews dataset, we explore how a single hidden layer neural network performs on binary classification tasks and analyze the trade-offs between model simplicity and performance.

The implementation showcases important machine learning concepts including data preprocessing, model training, validation, and the critical issue of overfitting. Through comprehensive visualization of training metrics, this project provides valuable insights into neural network behavior and serves as an educational foundation for understanding deep learning principles.

**Why This Project Matters:**
- 🧠 **Educational Value**: Perfect introduction to neural networks and deep learning concepts
- 📊 **Real-World Application**: Sentiment analysis is widely used in business intelligence and social media monitoring
- 🔍 **Overfitting Analysis**: Clear demonstration of overfitting patterns and validation techniques
- 🚀 **Foundation Building**: Establishes groundwork for more complex architectures

## Overview

This repository contains a comprehensive implementation of a single hidden layer neural network designed to tackle binary sentiment classification. The project systematically explores how minimal network architecture affects model performance, making it an ideal learning resource for understanding the fundamentals of deep learning.

## What You'll Learn

Through this project, you'll gain hands-on experience with:

- **Neural Network Fundamentals**: Understanding basic architecture components and their roles
- **Text Processing for ML**: Converting raw text data into numerical representations
- **Binary Classification**: Implementing and evaluating classification models
- **Overfitting Recognition**: Identifying and understanding overfitting through validation curves
- **Model Evaluation**: Using proper metrics and validation techniques
- **Data Visualization**: Creating meaningful plots to interpret model performance

## Project Workflow

1. **Data Loading**: Import and explore the IMDB dataset
2. **Preprocessing**: Convert text reviews into numerical vectors
3. **Model Architecture**: Build a simple 3-layer neural network
4. **Training**: Train with validation monitoring
5. **Evaluation**: Assess performance and analyze results
6. **Visualization**: Plot training curves to understand model behavior

## Project Description
The project implements a simple feedforward neural network with just one hidden layer to classify movie reviews as positive or negative sentiment. This minimal architecture serves as a baseline to understand the relationship between network complexity and performance.

## Dataset
- **Source**: IMDB Movie Reviews Dataset (via Keras)
- **Size**: 50,000 reviews total
  - 25,000 training samples
  - 25,000 test samples
- **Vocabulary**: Limited to top 10,000 most frequent words
- **Task**: Binary classification (positive/negative sentiment)

## Neural Network Architecture
- **Input Layer**: 10,000 neurons (one-hot encoded word vectors)
- **Hidden Layer**: 16 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)
- **Total Parameters**: Minimal architecture for computational efficiency

## Key Features
- **Data Preprocessing**: Custom vectorization function to convert word sequences to binary vectors
- **Validation Split**: 10,000 samples reserved for validation during training
- **Training Configuration**:
  - Optimizer: RMSprop
  - Loss Function: Binary crossentropy
  - Metrics: Accuracy
  - Epochs: 20 (with early stopping analysis)
  - Batch Size: 512

## Results
- **Final Test Accuracy**: 86.06%
- **Training Behavior**: Shows clear overfitting after epoch 4-5
- **Validation Performance**: Peaks around 88-89% before declining due to overfitting

## Visualizations
The project includes comprehensive plotting of:
- Training vs Validation Loss curves
- Training vs Validation Accuracy curves
- Clear demonstration of overfitting patterns

## Technical Implementation
- **Framework**: Keras/TensorFlow
- **Language**: Python
- **Key Libraries**: NumPy, Matplotlib, Keras
- **Environment**: Google Colab compatible

## Learning Objectives
1. Understanding the impact of network depth on performance
2. Recognizing overfitting patterns in training curves
3. Implementing basic neural network architectures
4. Data preprocessing for NLP tasks
5. Model evaluation and validation techniques

## File Structure
```
├── README.md                 # Project overview
└── Untitled1.ipynb         # Main implementation notebook
```

## Getting Started

### Prerequisites
- Python 3.6+
- TensorFlow/Keras
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sourav-maji/Deep-Learing-Project-1.git
   cd Deep-Learing-Project-1
   ```

2. Install required packages:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Untitled1.ipynb
   ```

Or simply click the "Open in Colab" badge above to run directly in Google Colab.

## Future Improvements
- Experiment with different hidden layer sizes
- Add multiple hidden layers for comparison
- Implement regularization techniques (dropout, L1/L2)
- Try different optimizers and learning rates
- Add more sophisticated text preprocessing

## Contributing
Feel free to fork this project and submit pull requests for any improvements!

## License
This project is open source and available under the [MIT License](LICENSE).

## Author
**Sourav Maji** - [GitHub Profile](https://github.com/sourav-maji)

---
*This project serves as an excellent starting point for understanding deep learning fundamentals and the importance of architecture choices in neural network design.*
