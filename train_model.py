
from app.core import milestone_4

print("=" * 50)
print("COURSE NOTES AI TUTOR - GPT TRAINING")
print("=" * 50)

result = milestone_4(
    epochs=30,
    learning_rate=0.003,
    block_size=16,
    embedding_dim=32,
    num_heads=4,
    batch_size=8
)

print("\nTRAINING SUMMARY")
print("-" * 50)

print("Vocabulary size:", result["vocab_size"])
print("Initial loss:", round(result["initial_loss"], 4))
print("Final loss:", round(result["final_loss"], 4))

parameters = sum(
    p.numel()
    for p in result["model"].parameters()
)

print("Model parameters:", parameters)

if result["final_loss"] < result["initial_loss"]:
    print("Training loss decreased overall.")
else:
    print("Loss fluctuated; try more training steps.")
