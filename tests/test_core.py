"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest


# Milestone 1: Prepare educational text corpus
def test_milestone_1():
    from app.core import milestone_1

    data = milestone_1()

    # Verify the returned data structure
    assert isinstance(data, dict)

    assert "notes" in data
    assert "corpus" in data
    assert "num_documents" in data
    assert "num_words" in data

    # Verify that 30 lecture notes are present
    assert data["num_documents"] == 30

    # Verify that the corpus is not empty
    assert len(data["corpus"]) > 0
    assert data["num_words"] > 100

    # Verify each note is cleaned
    for note in data["notes"]:
        assert isinstance(note, str)
        assert len(note) > 0
        assert note == note.lower()
        assert "  " not in note

    # Verify important educational concepts
    corpus = data["corpus"]

    assert "machine learning" in corpus
    assert "neural networks" in corpus
    assert "transformers" in corpus
    assert "tokenization" in corpus

    # Verify word count
    assert data["num_words"] == len(corpus.split())



# Milestone 2: BPE tokenizer and token embeddings
def test_milestone_2():
    import torch
    from app.core import milestone_1, milestone_2

    model = milestone_2(
        vocab_size=300,
        embedding_dim=32
    )

    # Verify tokenizer and embedding structure
    assert isinstance(model, dict)

    assert "vocab" in model
    assert "merges" in model
    assert "encode" in model
    assert "decode" in model
    assert "embedding" in model
    assert "embedded_tokens" in model

    # Verify vocabulary size
    assert 256 <= model["vocab_size"] <= 300

    # Verify that BPE learned at least one merge
    assert len(model["merges"]) > 0

    # Verify encoding and decoding
    sample = "machine learning"

    encoded = model["encode"](sample)
    decoded = model["decode"](encoded)

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert decoded == sample

    # Verify UTF-8 round trip
    unicode_sample = "AI tutor café"
    assert model["decode"](
        model["encode"](unicode_sample)
    ) == unicode_sample

    # Verify corpus round trip
    corpus = milestone_1()["corpus"]
    assert model["decode"](model["token_ids"]) == corpus

    # Verify embedding dimensions
    embeddings = model["embedded_tokens"]

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 32
    assert embeddings.shape[0] == len(
        model["token_ids"]
    )

    # Verify embedding parameters are trainable
    assert model["embedding"].weight.requires_grad




# Milestone 3: Causal self-attention and transformer
def test_milestone_3():
    import torch
    import pytest
    from app.core import milestone_2, milestone_3

    torch.manual_seed(42)

    model = milestone_3(
        embedding_dim=32,
        num_heads=4,
        block_size=64
    )

    transformer = model["transformer"]
    attention = model["attention"]

    # Verify model configuration
    assert model["embedding_dim"] == 32
    assert model["num_heads"] == 4
    assert model["block_size"] == 64

    # Verify transformer output shape
    x = torch.randn(2, 10, 32)
    output = transformer(x)

    assert output.shape == (2, 10, 32)
    assert torch.isfinite(output).all()

    # Verify attention output shape
    attention_output = attention(x)

    assert attention_output.shape == x.shape

    # Verify causal masking
    transformer.eval()

    original = torch.randn(1, 8, 32)
    modified = original.clone()

    # Change only future tokens
    modified[:, 4:, :] += 100.0

    with torch.no_grad():
        out1 = transformer(original)
        out2 = transformer(modified)

    # Earlier positions must remain unchanged
    assert torch.allclose(
        out1[:, :4, :],
        out2[:, :4, :],
        atol=1e-5
    )

    # Verify gradients flow through the model
    transformer.train()

    x = torch.randn(2, 10, 32)
    loss = transformer(x).sum()
    loss.backward()

    assert any(
        p.grad is not None
        for p in transformer.parameters()
    )

    # Verify compatibility with Milestone 2
    tokenizer = milestone_2(
        vocab_size=300,
        embedding_dim=32
    )

    tokens = tokenizer["token_ids"][:10]
    token_tensor = torch.tensor(
        tokens, dtype=torch.long
    ).unsqueeze(0)

    embeddings = tokenizer["embedding"](token_tensor)
    result = transformer(embeddings)

    assert result.shape == (1, 10, 32)

    # Invalid configurations should fail
    with pytest.raises(ValueError):
        milestone_3(embedding_dim=30, num_heads=4)

    with pytest.raises(ValueError):
        milestone_3(block_size=0)




# Milestone 4: Train GPT using gradient descent
def test_milestone_4():
    import torch
    import pytest
    from app.core import milestone_4

    # Train a small model for testing
    result = milestone_4(
        epochs=5,
        learning_rate=0.003,
        block_size=16,
        embedding_dim=32,
        num_heads=4,
        batch_size=4
    )

    # Verify returned training information
    assert isinstance(result, dict)
    assert "model" in result
    assert "tokenizer" in result
    assert "loss_history" in result
    assert "initial_loss" in result
    assert "final_loss" in result

    # Verify training steps
    losses = result["loss_history"]

    assert len(losses) == 5
    assert all(
        isinstance(loss, float)
        for loss in losses
    )

    # Verify losses are finite and positive
    assert all(
        0 < loss < float("inf")
        for loss in losses
    )

    # Verify model forward pass
    model = result["model"]
    model.eval()

    token_ids = result["tokenizer"]["token_ids"][:8]

    x = torch.tensor(
        [token_ids], dtype=torch.long
    )

    with torch.no_grad():
        logits, loss = model(x, x)

    assert logits.shape == (
        1, 8, result["vocab_size"]
    )

    assert loss is not None
    assert torch.isfinite(loss)

    # Verify model has trainable parameters
    assert sum(
        p.numel() for p in model.parameters()
    ) > 0

    # Invalid training configuration
    with pytest.raises(ValueError):
        milestone_4(epochs=0)

    with pytest.raises(ValueError):
        milestone_4(learning_rate=-0.1)




# Milestone 5: Generate text and save checkpoints
def test_milestone_5(tmp_path):
    import pytest
    from app.core import milestone_5

    checkpoint = tmp_path / "tiny_gpt.pt"

    # Train and generate text
    result = milestone_5(
        prompt="machine learning is",
        max_new_tokens=5,
        temperature=0.8,
        checkpoint_path=str(checkpoint),
        epochs=3
    )

    assert isinstance(result, dict)
    assert checkpoint.exists()

    assert "generated_text" in result
    assert "generated_token_ids" in result

    assert result["generated_text"].startswith(
        "machine learning is"
    )

    # Generated IDs include the original prompt
    assert len(result["generated_token_ids"]) > 5

    # Load the existing checkpoint
    loaded = milestone_5(
        prompt="artificial intelligence",
        max_new_tokens=3,
        checkpoint_path=str(checkpoint),
        train_if_missing=False
    )

    assert loaded["generated_text"].startswith(
        "artificial intelligence"
    )

    # Validate incorrect input
    with pytest.raises(ValueError):
        milestone_5(prompt="")

    with pytest.raises(ValueError):
        milestone_5(max_new_tokens=0)

    with pytest.raises(ValueError):
        milestone_5(temperature=0)

    with pytest.raises(FileNotFoundError):
        milestone_5(
            checkpoint_path=str(
                tmp_path / "missing.pt"
            ),
            train_if_missing=False
        )


# Milestone 6: Create a simple CLI interface for interacting with the AI tutor.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

