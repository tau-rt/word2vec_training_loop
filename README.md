# word2vec_training_loop

Pure NumPy implementation of Word2Vec using the Skip-gram model with negative sampling. The project includes text preprocessing, training pair generation, manual gradient computation, and embedding evaluation via cosine similarity.

---

## Description

This project implements Word2Vec from scratch in pure NumPy, without using any machine learning frameworks such as PyTorch, TensorFlow, or gensim.

The model is based on **skip-gram with Negative Sampling**. It learns word embeddings by predicting context words given a center word using a sliding window (size of 2 by default).

The implementation includes:
- text preprocessing and tokenization
- stopword removal
- vocabulary construction
- generation of (center, context) pairs
- frequency-based negative sampling (`count^0.75`)
- forward pass using dot product and sigmoid
- binary cross-entropy loss
- manual gradient computation
- parameter updates using gradient descent

The model is trained on a cleaned version of *Crime and Punishment* and evaluated using cosine similarity between word embeddings.

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

---

## Dataset

The model is trained on a cleaned plain-text version of *Crime and Punishment*.
The dataset is included in the repository.

## Example Output:

Most similar words:
raskolnikov → razumikhin, sonia, …
crime → murder, conscience, …

Pair similarities:
similarity(crime, murder) = 0.93
similarity(prison, punishment) = 0.98
similarity(crime, table) = 0.74
