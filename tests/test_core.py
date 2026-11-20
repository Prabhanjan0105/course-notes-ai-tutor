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



# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


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

