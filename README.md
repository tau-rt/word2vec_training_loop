# word2vec_training_loop

Pure NumPy implementation of Word2Vec using the Skip-gram model with negative sampling. The project includes text preprocessing, training pair generation, manual gradient computation, and embedding evaluation via cosine similarity.

---

## Description

This project implements Word2Vec from scratch in pure NumPy, without using machine learning frameworks such as PyTorch, TensorFlow, or gensim.

The model is based on the **Skip-gram architecture with Negative Sampling (SGNS)**. It learns word embeddings by predicting context words given a center word using a sliding window.

The implementation includes:
- text preprocessing and tokenization
- stopword removal
- filtering rare words (`min_count`)
- vocabulary construction
- generation of (center, context) pairs
- frequency-based negative sampling (`count^0.75`)
- forward pass using dot product and sigmoid
- binary cross-entropy loss
- manual gradient computation
- parameter updates using gradient descent

The model is trained on a cleaned version of *Crime and Punishment* and evaluated using cosine similarity between word embeddings.

---

## Preprocessing

The text is processed as follows:
- converted to lowercase
- split into sentences
- punctuation removed (keeping only `a–z`)
- stopwords removed
- tokens shorter than 3 characters removed
- rare words filtered using `min_count`

This reduces noise and improves embedding quality.

---

## Training Procedure

For each (center, context) pair:

1. Compute dot product between embeddings
2. Apply sigmoid activation
3. Compute binary cross-entropy loss:
   - positive sample
   - negative samples
4. Compute gradients manually
5. Update embeddings using gradient descent

---

## Hyperparameters (default)

| Parameter            | Value |
|---------------------|-------|
| Window size         | 2     |
| Embedding dimension | 20    |
| Epochs              | 10    |
| Learning rate       | 0.02  |
| Negative samples    | 5     |
| Min word frequency  | 5     |

---

## Dataset

The model is trained on a cleaned plain-text version of *Crime and Punishment*.
