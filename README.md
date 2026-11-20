# Course Notes AI Tutor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dzsntexgyg/badge.svg)](https://ask.thili.ai/b/dzsntexgyg)

This project, 'Course Notes AI Tutor', challenges you to build a simple command-line tutor that learns from student course notes. It will complete sentences and offer short explanations on concepts to aid revision. You will apply your knowledge of building GPT from scratch, training the model on raw educational text to learn contextual relationships and generate relevant sentence completions.

## What you are building

**Goal.** You will build a command-line AI tutor that learns from course notes, completing sentences and offering short explanations to aid student revision.

**Done when.** The project is done when the AI tutor can prepare raw text using a byte-pair tokenizer and embeddings, implement causal self-attention, assemble a transformer block, and train a GPT model from scratch to generate coherent text completions and explanations.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small corpus of educational text data (e.g., sample lecture notes).**  
      Hint: Data preparation and cleaning
- [ ] **2. Implement the byte-pair tokenizer and generate token embeddings from the corpus.**  
      Hint: Byte-pair encoding and token embeddings
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal self-attention and transformer blocks
- [ ] **4. Set up the training loop and train the GPT model on the prepared text.**  
      Hint: Training loop and gradient descent
- [ ] **5. Implement text generation, allowing users to input a prompt and get completions.**  
      Hint: Text generation with sampling
- [ ] **6. Create a simple CLI interface for interacting with the AI tutor.**  
      Hint: Command-line interface (CLI) development

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
