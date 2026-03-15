Word2Vec (NumPy Implementation)

This repository contains a minimal implementation of the Word2Vec algorithm written in pure NumPy.
The goal of the project is to demonstrate a full understanding of the training process behind
Word2Vec, including preprocessing, forward propagation, loss computation, gradient calculation,
and parameter updates.

The implementation follows the skip-gram formulation using a simple neural network with a
softmax output layer. A center word is used to predict surrounding context words within a
fixed window. The model learns word embeddings by training two weight matrices that map
words to a lower-dimensional vector space.

The project is structured into three main components.

preprocessing.py handles tokenization, vocabulary creation, one-hot encoding, and generation
of training pairs based on a context window.

training.py implements the neural network model, softmax activation, cross-entropy loss,
forward pass, backpropagation, and gradient descent updates.

main.py loads a text dataset, prepares training data, runs the training loop, visualizes
the loss curve, and allows simple inspection of learned embeddings and word predictions.

The neural network consists of an input layer represented as a one-hot encoded vector,
an embedding layer defined by the matrix W1, and an output layer defined by W2. The
forward pass computes the embedding of the input word and predicts probabilities over
the vocabulary using softmax. During training, gradients are computed analytically and
the parameters are updated using gradient descent.

A small text corpus is included for demonstration purposes, but any plain text file can
be used as training data by replacing text.txt.

To run the training script:

python main.py
