from __future__ import annotations

import argparse
import random
from pathlib import Path


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic long-range text corpora for CA-vs-transformer experiments."
    )
    parser.add_argument(
        "--task",
        choices=("copy", "delayed_copy", "brackets", "induction"),
        required=True,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--delay-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def random_payload(rng: random.Random, min_length: int, max_length: int) -> str:
    payload_length = rng.randint(min_length, max_length)
    return "".join(rng.choice(ALPHABET[:8]) for _ in range(payload_length))


def generate_copy_example(
    rng: random.Random,
    min_length: int,
    max_length: int,
) -> str:
    payload = random_payload(rng, min_length=min_length, max_length=max_length)
    return f"copy:{payload}|{payload}\n"


def generate_delayed_copy_example(
    rng: random.Random,
    min_length: int,
    max_length: int,
    delay_length: int,
) -> str:
    payload = random_payload(rng, min_length=min_length, max_length=max_length)
    filler = "".join(rng.choice("xyz") for _ in range(delay_length))
    return f"delay:{payload}|{filler}|{payload}\n"


def balanced_bracket_string(rng: random.Random, num_pairs: int) -> str:
    opens: list[str] = []
    output: list[str] = []
    for _ in range(num_pairs):
        open_bracket, close_bracket = rng.choice(BRACKET_PAIRS)
        output.append(open_bracket)
        opens.append(close_bracket)
        while opens and rng.random() < 0.4:
            output.append(opens.pop())
    while opens:
        output.append(opens.pop())
    return "".join(output)


def unbalance_bracket_string(rng: random.Random, sequence: str) -> str:
    if len(sequence) <= 1:
        return sequence + "]"
    mutation_index = rng.randrange(len(sequence))
    replacement = rng.choice("()[]{}")
    while replacement == sequence[mutation_index]:
        replacement = rng.choice("()[]{}")
    return sequence[:mutation_index] + replacement + sequence[mutation_index + 1 :]


def generate_bracket_example(
    rng: random.Random,
    min_length: int,
    max_length: int,
) -> str:
    num_pairs = max(1, rng.randint(max(1, min_length // 2), max(1, max_length // 2)))
    balanced = balanced_bracket_string(rng, num_pairs=num_pairs)
    is_balanced = rng.random() < 0.5
    sequence = balanced if is_balanced else unbalance_bracket_string(rng, balanced)
    label = "1" if is_balanced else "0"
    return f"brackets:{sequence}={label}\n"


def generate_induction_example(
    rng: random.Random,
    min_length: int,
    max_length: int,
) -> str:
    num_pairs = max(3, rng.randint(min_length // 2, max_length // 2))
    keys = rng.sample(ALPHABET[:12], k=min(num_pairs, 12))
    values = rng.sample(ALPHABET[12:], k=len(keys))
    pairs = list(zip(keys, values))
    query_key, query_value = rng.choice(pairs)
    memory = ";".join(f"{key}>{value}" for key, value in pairs)
    return f"induct:{memory}?{query_key}={query_value}\n"


def generate_corpus(
    task: str,
    num_examples: int,
    min_length: int,
    max_length: int,
    delay_length: int,
    seed: int,
) -> str:
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_examples):
        if task == "copy":
            lines.append(
                generate_copy_example(
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                )
            )
        elif task == "delayed_copy":
            lines.append(
                generate_delayed_copy_example(
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                    delay_length=delay_length,
                )
            )
        elif task == "brackets":
            lines.append(
                generate_bracket_example(
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                )
            )
        elif task == "induction":
            lines.append(
                generate_induction_example(
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                )
            )
        else:
            raise ValueError(f"Unsupported synthetic task: {task}")
    return "".join(lines)


def write_synthetic_corpus(
    task: str,
    output_path: Path,
    num_examples: int,
    min_length: int,
    max_length: int,
    delay_length: int,
    seed: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = generate_corpus(
        task=task,
        num_examples=num_examples,
        min_length=min_length,
        max_length=max_length,
        delay_length=delay_length,
        seed=seed,
    )
    output_path.write_text(corpus, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    write_synthetic_corpus(
        task=args.task,
        output_path=args.output_path,
        num_examples=args.num_examples,
        min_length=args.min_length,
        max_length=args.max_length,
        delay_length=args.delay_length,
        seed=args.seed,
    )
    print(f"saved synthetic corpus to: {args.output_path}")


if __name__ == "__main__":
    main()
