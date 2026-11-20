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




def milestone_3(embedding_dim=32, num_heads=4,
                block_size=64, dropout=0.0):
    """
    Build causal multi-head self-attention
    and a GPT-style transformer block.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if embedding_dim <= 0 or num_heads <= 0:
        raise ValueError(
            "Embedding dimension and heads must be positive"
        )

    if embedding_dim % num_heads != 0:
        raise ValueError(
            "embedding_dim must be divisible by num_heads"
        )

    if block_size < 1:
        raise ValueError("block_size must be positive")

    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()

            self.num_heads = num_heads
            self.head_dim = embedding_dim // num_heads

            self.qkv = nn.Linear(
                embedding_dim, 3 * embedding_dim
            )

            self.proj = nn.Linear(
                embedding_dim, embedding_dim
            )

            self.attn_dropout = nn.Dropout(dropout)
            self.resid_dropout = nn.Dropout(dropout)

            mask = torch.tril(
                torch.ones(block_size, block_size)
            )

            self.register_buffer(
                "mask",
                mask.view(1, 1, block_size, block_size)
            )

        def forward(self, x):
            B, T, C = x.shape

            if T > block_size:
                raise ValueError(
                    "Sequence exceeds block_size"
                )

            qkv = self.qkv(x)

            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(
                B, T, self.num_heads, self.head_dim
            ).transpose(1, 2)

            k = k.view(
                B, T, self.num_heads, self.head_dim
            ).transpose(1, 2)

            v = v.view(
                B, T, self.num_heads, self.head_dim
            ).transpose(1, 2)

            # Scaled dot-product attention
            scores = (
                q @ k.transpose(-2, -1)
            ) / (self.head_dim ** 0.5)

            # Prevent attention to future tokens
            scores = scores.masked_fill(
                self.mask[:, :, :T, :T] == 0,
                float("-inf")
            )

            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)

            output = weights @ v

            output = output.transpose(
                1, 2
            ).contiguous().view(B, T, C)

            return self.resid_dropout(
                self.proj(output)
            )

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()

            self.ln1 = nn.LayerNorm(embedding_dim)
            self.attention = CausalSelfAttention()

            self.ln2 = nn.LayerNorm(embedding_dim)

            self.feed_forward = nn.Sequential(
                nn.Linear(
                    embedding_dim,
                    4 * embedding_dim
                ),
                nn.GELU(),
                nn.Linear(
                    4 * embedding_dim,
                    embedding_dim
                ),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            # Attention with residual connection
            x = x + self.attention(self.ln1(x))

            # Feed-forward with residual connection
            x = x + self.feed_forward(self.ln2(x))

            return x

    transformer = TransformerBlock()

    return {
        "transformer": transformer,
        "attention": transformer.attention,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "block_size": block_size
    }




def milestone_4(epochs=30, learning_rate=0.003,
                block_size=16, embedding_dim=32,
                num_heads=4, batch_size=8):
    """
    Build and train a small GPT-style model
    on the educational text corpus.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be positive")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(42)

    # Milestone 2: Train BPE tokenizer
    tokenizer = milestone_2(
        vocab_size=300,
        embedding_dim=embedding_dim
    )

    vocab_size = tokenizer["vocab_size"]

    data = torch.tensor(
        tokenizer["token_ids"],
        dtype=torch.long
    )

    if len(data) <= block_size:
        raise ValueError(
            "Corpus must be longer than block_size"
        )

    # Milestone 3: Transformer architecture
    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()

            # Reuse token embeddings from Milestone 2
            self.token_embedding = tokenizer["embedding"]

            self.position_embedding = nn.Embedding(
                block_size, embedding_dim
            )

            self.blocks = nn.Sequential(
                milestone_3(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size
                )["transformer"],
                milestone_3(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size
                )["transformer"]
            )

            self.ln_final = nn.LayerNorm(
                embedding_dim
            )

            self.lm_head = nn.Linear(
                embedding_dim,
                vocab_size
            )

        def forward(self, idx, targets=None):
            B, T = idx.shape

            if T > block_size:
                raise ValueError(
                    "Input exceeds block_size"
                )

            positions = torch.arange(
                T, device=idx.device
            )

            tok_emb = self.token_embedding(idx)
            pos_emb = self.position_embedding(
                positions
            )

            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_final(x)

            logits = self.lm_head(x)

            loss = None

            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1)
                )

            return logits, loss

    model = TinyGPT()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    def get_batch():
        # Sample training windows from corpus
        starts = torch.randint(
            0,
            len(data) - block_size,
            (batch_size,)
        )

        x = torch.stack([
            data[i:i + block_size]
            for i in starts.tolist()
        ])

        y = torch.stack([
            data[i + 1:i + block_size + 1]
            for i in starts.tolist()
        ])

        return x, y

    loss_history = []

    print("\nStarting GPT model training...")
    print("-" * 45)

    model.train()

    for epoch in range(epochs):
        x, y = get_batch()

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == epochs - 1
        ):
            print(
                f"Step {epoch + 1}/{epochs} "
                f"| Loss: {loss_value:.4f}"
            )

    model.eval()

    print("-" * 45)
    print("Training completed!")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "vocab_size": vocab_size,
        "block_size": block_size,
        "embedding_dim": embedding_dim
    }




def milestone_5(prompt="machine learning is",
                max_new_tokens=30,
                temperature=0.8,
                checkpoint_path="checkpoints/tiny_gpt.pt",
                train_if_missing=True,
                epochs=30):
    """
    Generate text using a trained GPT model.
    Save and load model checkpoints.
    """
    from pathlib import Path
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    checkpoint = Path(checkpoint_path)

    # Configuration must match the training model.
    config = {
        "block_size": 16,
        "embedding_dim": 32,
        "num_heads": 4,
        "batch_size": 8,
        "learning_rate": 0.003
    }

    if checkpoint.exists():
        print("Loading saved GPT checkpoint...")

        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )

        config = saved["config"]

        # Reconstruct the same architecture.
        # One initialization step is used because
        # milestone_4 currently builds the model
        # inside its training function.
        result = milestone_4(
            epochs=1,
            **config
        )

        model = result["model"]
        model.load_state_dict(saved["model_state_dict"])

        tokenizer = result["tokenizer"]

        print("Checkpoint loaded successfully!")

    else:
        if not train_if_missing:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        print("No checkpoint found. Training model...")

        result = milestone_4(
            epochs=epochs,
            **config
        )

        model = result["model"]
        tokenizer = result["tokenizer"]

        checkpoint.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config
        }, checkpoint)

        print(f"Checkpoint saved to: {checkpoint}")

    model.eval()

    # Encode the student prompt
    token_ids = tokenizer["encode"](prompt)

    if not token_ids:
        raise ValueError("Prompt produced no tokens")

    generated = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    # Generate one token at a time
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Keep only the latest context window
            context = generated[
                :, -config["block_size"]:
            ]

            logits, _ = model(context)

            # Take logits for the next token
            next_logits = logits[:, -1, :]

            # Apply temperature
            next_logits = next_logits / temperature

            probabilities = torch.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

    # Decode generated token IDs
    # Use replacement for incomplete UTF-8 sequences.
    generated_ids = generated[0].tolist()

    raw_bytes = b"".join(
        tokenizer["vocab"][token_id]
        for token_id in generated_ids
    )

    generated_text = raw_bytes.decode(
        "utf-8",
        errors="replace"
    )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "checkpoint_path": str(checkpoint),
        "temperature": temperature
    }




def milestone_6(prompt="machine learning is",
                max_new_tokens=30,
                temperature=0.8,
                checkpoint_path="checkpoints/tiny_gpt.pt",
                train_if_missing=True,
                epochs=30):
    """
    CLI-facing interface for the Course Notes AI Tutor.

    Uses the text generation implementation
    developed in Milestone 5.
    """
    result = milestone_5(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        checkpoint_path=checkpoint_path,
        train_if_missing=train_if_missing,
        epochs=epochs
    )

    return {
        "prompt": result["prompt"],
        "response": result["generated_text"],
        "token_ids": result["generated_token_ids"],
        "checkpoint_path": result["checkpoint_path"],
        "temperature": result["temperature"]
    }

