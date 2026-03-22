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

## Training Procedure

For each (center, context) pair:

1. Compute dot product between embeddings
2. Apply sigmoid activation
3. Compute binary cross-entropy loss:
   - positive sample
   - negative samples
4. Compute gradients manually
5. Update embeddings using gradient descent

## Hyperparameters (default)

| Parameter            | Value |
|---------------------|-------|
| Window size         | 2     |
| Embedding dimension | 20    |
| Epochs              | 10    |
| Learning rate       | 0.02  |
| Negative samples    | 5     |

## Limitations
- Small corpus leads to limited semantic separation
- Uniform negative sampling (not frequency-based)
- No subsampling of frequent words

