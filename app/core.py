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



# --- Milestone 2: Implement the byte-pair tokenizer and generate token embeddings from the corpus.
# Hint: Byte-pair encoding and token embeddings
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
