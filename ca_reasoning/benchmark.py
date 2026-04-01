from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import build_vocab, read_text
from .training import ModelConfig, benchmark_training_step, maybe_prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark training-step time and memory for transformer and CA models."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("transformer", "ca"),
        default=["transformer", "ca"],
    )

    parser.add_argument("--transformer-hidden-dim", type=int, default=96)
    parser.add_argument("--transformer-num-layers", type=int, default=4)
    parser.add_argument("--transformer-num-heads", type=int, default=4)

    parser.add_argument("--ca-hidden-dim", type=int, default=96)
    parser.add_argument("--ca-steps", type=int, default=8)
    parser.add_argument("--ca-rule-sharing", choices=("shared", "unshared"), default="unshared")
    parser.add_argument(
        "--ca-grid-layout",
        choices=("row_major_2d", "tape_1d"),
        default="row_major_2d",
    )
    parser.add_argument(
        "--ca-neighborhood",
        choices=("3x3_masked", "5x5_masked", "3x3_dilated"),
        default="3x3_masked",
    )
    parser.add_argument(
        "--ca-position-mode",
        choices=("seq_only", "grid_only", "both", "none"),
        default="both",
    )

    parser.add_argument("--results-path", type=Path, default=Path("results/benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = maybe_prepare_data(args.data_path, should_download=args.download_dataset)
    vocab = build_vocab(read_text(data_path))

    benchmarks: list[dict[str, object]] = []
    for context_len in args.contexts:
        if "transformer" in args.models:
            model_config = ModelConfig(
                model="transformer",
                context_len=context_len,
                hidden_dim=args.transformer_hidden_dim,
                num_layers=args.transformer_num_layers,
                num_heads=args.transformer_num_heads,
            )
            result = benchmark_training_step(
                model_config=model_config,
                vocab_size=vocab.size,
                batch_size=args.batch_size,
                device_name=args.device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                seed=args.seed,
            )
            benchmarks.append(
                {
                    "model": "transformer",
                    "context_len": context_len,
                    "model_config": asdict(model_config),
                    "benchmark": result,
                }
            )
            print(
                f"transformer context={context_len} "
                f"mean_step_ms={float(result['mean_step_ms']):.3f} "
                f"peak_memory_mb={result['peak_memory_mb']}"
            )

        if "ca" in args.models:
            model_config = ModelConfig(
                model="ca",
                context_len=context_len,
                hidden_dim=args.ca_hidden_dim,
                ca_steps=args.ca_steps,
                rule_sharing=args.ca_rule_sharing,
                grid_layout=args.ca_grid_layout,
                neighborhood=args.ca_neighborhood,
                position_mode=args.ca_position_mode,
            )
            result = benchmark_training_step(
                model_config=model_config,
                vocab_size=vocab.size,
                batch_size=args.batch_size,
                device_name=args.device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                seed=args.seed,
            )
            benchmarks.append(
                {
                    "model": "ca",
                    "context_len": context_len,
                    "model_config": asdict(model_config),
                    "benchmark": result,
                }
            )
            print(
                f"ca context={context_len} "
                f"mean_step_ms={float(result['mean_step_ms']):.3f} "
                f"peak_memory_mb={result['peak_memory_mb']}"
            )

    payload = {
        "kind": "benchmark",
        "data_path": str(data_path),
        "vocab_size": vocab.size,
        "benchmarks": benchmarks,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved benchmark results to: {args.results_path}")


if __name__ == "__main__":
    main()
