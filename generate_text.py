
from app.core import milestone_5

print("=" * 55)
print("COURSE NOTES AI TUTOR - TEXT GENERATION")
print("=" * 55)

result = milestone_5(
    prompt="machine learning is",
    max_new_tokens=30,
    temperature=0.8,
    epochs=30
)

print("\nSTUDENT PROMPT:")
print(result["prompt"])

print("\nGENERATED TEXT:")
print(result["generated_text"])

print("\nCheckpoint:")
print(result["checkpoint_path"])

print("=" * 55)
