# word2vec_training_loop
Pure NumPy implementation of Word2Vec using the Skip-gram model with negative sampling. The project includes text preprocessing, training pair generation, manual gradient computation, and embedding evaluation via cosine similarity.


## Description

This project implements Word2Vec from scratch in pure NumPy, without using any machine learning frameworks such as PyTorch, TensorFlow, or gensim.

The model is based on the Skip-gram architecture with negative sampling. It learns word embeddings by predicting context words given a center word using a sliding window approach.

The implementation includes:
- text preprocessing and tokenization
- vocabulary construction
- generation of (center, context) pairs
- negative sampling
- forward pass using dot product and sigmoid
- binary cross-entropy loss
- manual gradient computation
- parameter updates using gradient descent

The model is trained on a cleaned version of *Crime and Punishment* and evaluated using cosine similarity between word embeddings.
