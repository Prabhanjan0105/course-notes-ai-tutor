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



# Milestone 4: Set up the training loop and train the GPT model on the prepared text.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement text generation, allowing users to input a prompt and get completions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a simple CLI interface for interacting with the AI tutor.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

