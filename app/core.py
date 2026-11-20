"""Your code. One stub per milestone; build them in order and keep each one small."""


def milestone_1():
    """
    Prepare and clean a small corpus of educational
    lecture notes for GPT model training.
    """
    import re

    lecture_notes = [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Artificial Intelligence enables computers to perform tasks that require human reasoning.",
        "Machine Learning is a branch of Artificial Intelligence that learns from data.",
        "Machine Learning algorithms identify patterns and make predictions.",
        "Supervised Learning uses labeled data to train predictive models.",
        "Unsupervised Learning discovers hidden patterns in unlabeled data.",
        "Reinforcement Learning trains agents through rewards and penalties.",
        "Deep Learning uses multiple layers of neural networks.",
        "Deep Learning models can learn complex representations from large datasets.",
        "Neural Networks are inspired by the structure of the human brain.",
        "A neural network consists of interconnected artificial neurons.",
        "Each neuron processes inputs using weights and activation functions.",
        "Backpropagation calculates gradients to update neural network weights.",
        "Gradient Descent minimizes the loss function during training.",
        "Natural Language Processing helps computers understand human language.",
        "Tokenization divides text into smaller units called tokens.",
        "Word embeddings represent words as numerical vectors.",
        "Transformers use attention mechanisms to process sequences.",
        "Self Attention allows tokens to interact with other tokens.",
        "Causal Attention prevents a model from seeing future tokens.",
        "GPT stands for Generative Pretrained Transformer.",
        "GPT models predict the next token based on previous tokens.",
        "Language Models learn probability distributions over text sequences.",
        "Training a language model involves minimizing prediction loss.",
        "The softmax function converts model scores into probabilities.",
        "The Adam optimizer updates model parameters using gradients.",
        "Overfitting occurs when a model memorizes training examples.",
        "Regularization helps improve the generalization of machine learning models.",
        "Evaluation measures how well a trained model performs.",
        "Text Generation produces new text using learned language patterns."
    ]

    cleaned_notes = []

    for note in lecture_notes:
        # Convert text to lowercase
        text = note.lower()

        # Remove unnecessary special characters
        text = re.sub(r"[^a-z0-9\s.,]", "", text)

        # Normalize extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            cleaned_notes.append(text)

    # Combine the notes into a training corpus
    corpus = "\n".join(cleaned_notes)

    return {
        "notes": cleaned_notes,
        "corpus": corpus,
        "num_documents": len(cleaned_notes),
        "num_words": len(corpus.split())
    }



def milestone_2(vocab_size=300, embedding_dim=32):
    """
    Train a byte-pair tokenizer and create token embeddings.
    """
    from collections import Counter
    import torch
    import torch.nn as nn

    if not isinstance(vocab_size, int) or vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")

    if not isinstance(embedding_dim, int) or embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")

    # Get educational corpus from Milestone 1
    data = milestone_1()
    corpus = data["corpus"]

    # UTF-8 bytes become initial token IDs (0 to 255)
    token_ids = list(corpus.encode("utf-8"))

    if not token_ids:
        raise ValueError("Training corpus cannot be empty")

    merges = {}
    vocab = {
        i: bytes([i]) for i in range(256)
    }

    def get_pairs(ids):
        return Counter(
            zip(ids, ids[1:])
        )

    def merge_pair(ids, pair, new_id):
        result = []
        i = 0

        while i < len(ids):
            if (
                i < len(ids) - 1
                and (ids[i], ids[i + 1]) == pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    # Learn the most frequent adjacent byte pairs
    for new_id in range(256, vocab_size):
        pairs = get_pairs(token_ids)

        if not pairs:
            break

        pair, count = max(
            pairs.items(),
            key=lambda item: item[1]
        )

        if count < 2:
            break

        token_ids = merge_pair(
            token_ids, pair, new_id
        )

        merges[pair] = new_id

        vocab[new_id] = (
            vocab[pair[0]] + vocab[pair[1]]
        )

    def encode(text):
        """Convert text into BPE token IDs."""
        ids = list(text.encode("utf-8"))

        # Apply learned merges in training order
        for pair, new_id in merges.items():
            ids = merge_pair(ids, pair, new_id)

        return ids

    def decode(ids):
        """Convert BPE token IDs back into text."""
        raw_bytes = b"".join(
            vocab[token_id] for token_id in ids
        )
        return raw_bytes.decode("utf-8")

    actual_vocab_size = len(vocab)

    # Trainable token embedding layer
    embedding = nn.Embedding(
        num_embeddings=actual_vocab_size,
        embedding_dim=embedding_dim
    )

    encoded_corpus = encode(corpus)

    token_tensor = torch.tensor(
        encoded_corpus,
        dtype=torch.long
    )

    embedded_tokens = embedding(token_tensor)

    return {
        "vocab_size": actual_vocab_size,
        "embedding_dim": embedding_dim,
        "vocab": vocab,
        "merges": merges,
        "encode": encode,
        "decode": decode,
        "token_ids": encoded_corpus,
        "embedding": embedding,
        "embedded_tokens": embedded_tokens
    }



# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal self-attention and transformer blocks
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Set up the training loop and train the GPT model on the prepared text.
# Hint: Training loop and gradient descent
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement text generation, allowing users to input a prompt and get completions.
# Hint: Text generation with sampling
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a simple CLI interface for interacting with the AI tutor.
# Hint: Command-line interface (CLI) development
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
