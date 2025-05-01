# MNIST Neural Network From Scratch

This project implements a simple feedforward neural network (multi-layer perceptron) from scratch in Python to classify handwritten digits from the MNIST dataset. The neural network uses numpy for matrix operations and pandas for data handling, and is trained using gradient descent with mini-batches.

## Features
- No deep learning frameworks required (no TensorFlow, PyTorch, etc.)
- Implements forward and backward propagation manually
- Classifies digits 0-9 from the MNIST dataset
- Achieves 90% accuracy on the test set

## Requirements
- Python 3.x
- numpy
- pandas

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Dataset
You need to download the MNIST dataset CSV files from [Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or another source. Place `mnist_train.csv` and `mnist_test.csv` in a `data/` folder in the project directory.

## Running the Program
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Download the MNIST CSV files and place them in `data/` as described above.
3. Run the neural network:
    ```bash
    python DigitNN.py
    ```

The script will train the neural network and print the test accuracy after each epoch.
