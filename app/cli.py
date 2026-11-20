
"""Command-line interface for Course Notes AI Tutor."""

import argparse


def generate_response(
    prompt,
    max_new_tokens=30,
    temperature=0.8,
    checkpoint_path="checkpoints/tiny_gpt.pt",
    train_if_missing=True,
    epochs=30
):
    """Call the GPT generation pipeline."""
    from app.core import milestone_6

    return milestone_6(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        checkpoint_path=checkpoint_path,
        train_if_missing=train_if_missing,
        epochs=epochs
    )


def interactive_mode(args):
    """Start a continuous student-tutor conversation."""
    print("\n" + "=" * 55)
    print("COURSE NOTES AI TUTOR")
    print("=" * 55)
    print("Enter a topic or sentence to generate a completion.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("Student > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a prompt.\n")
            continue

        try:
            result = generate_response(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                checkpoint_path=args.checkpoint,
                train_if_missing=not args.no_train,
                epochs=args.epochs
            )

            print("\nAI Tutor >")
            print(result["response"])
            print()

        except (ValueError, FileNotFoundError) as error:
            print(f"Error: {error}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="course-notes-ai-tutor",
        description=(
            "A small GPT-style tutor trained "
            "on educational lecture notes."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser(
        "hello",
        help="Check that the tutor is ready"
    )

    sub.add_parser(
        "info",
        help="Display information about the project"
    )

    generate_parser = sub.add_parser(
        "generate",
        help="Generate text from a student prompt"
    )

    interactive_parser = sub.add_parser(
        "interactive",
        help="Start interactive tutor mode"
    )

    for command_parser in (
        generate_parser,
        interactive_parser
    ):
        command_parser.add_argument(
            "--max-tokens",
            type=int,
            default=30,
            help="Number of new tokens to generate"
        )

        command_parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Sampling temperature"
        )

        command_parser.add_argument(
            "--checkpoint",
            default="checkpoints/tiny_gpt.pt",
            help="Path to saved model checkpoint"
        )

        command_parser.add_argument(
            "--no-train",
            action="store_true",
            help="Do not train if checkpoint is missing"
        )

        command_parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Training steps if checkpoint is missing"
        )

    generate_parser.add_argument(
        "--prompt",
        required=True,
        help="Student prompt for text generation"
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Course Notes AI Tutor is ready!")

    elif args.command == "info":
        print("\nCOURSE NOTES AI TUTOR")
        print("-" * 40)
        print("Architecture: GPT-style Transformer")
        print("Tokenizer: Byte-Pair Encoding")
        print("Framework: PyTorch")
        print("Attention: Causal Self-Attention")
        print("Training: AdamW + Cross-Entropy")
        print("Interface: Command-Line")
        print("-" * 40)

    elif args.command == "generate":
        try:
            result = generate_response(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                checkpoint_path=args.checkpoint,
                train_if_missing=not args.no_train,
                epochs=args.epochs
            )

            print("\nSTUDENT PROMPT:")
            print(result["prompt"])

            print("\nAI TUTOR RESPONSE:")
            print(result["response"])

        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))

    elif args.command == "interactive":
        interactive_mode(args)


if __name__ == "__main__":
    main()
