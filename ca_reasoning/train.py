from __future__ import annotations

import argparse
from pathlib import Path

from .training import ModelConfig, TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transformer or cellular automata language model."
    )
    parser.add_argument("--model", choices=("transformer", "ca"), default="ca")
    parser.add_argument("--data-path", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ca-steps", type=int, default=8)
    parser.add_argument(
        "--rule-sharing",
        choices=("shared", "unshared"),
        default="unshared",
    )
    parser.add_argument(
        "--grid-layout",
        choices=("row_major_2d", "tape_1d"),
        default="row_major_2d",
    )
    parser.add_argument(
        "--ca-neighborhood",
        choices=("3x3_masked", "5x5_masked", "3x3_dilated"),
        default="3x3_masked",
    )
    parser.add_argument(
        "--position-mode",
        choices=("seq_only", "grid_only", "both", "none"),
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(
        model=args.model,
        context_len=args.context_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ca_steps=args.ca_steps,
        rule_sharing=args.rule_sharing,
        grid_layout=args.grid_layout,
        neighborhood=args.ca_neighborhood,
        position_mode=args.position_mode,
    )
    train_config = TrainConfig(
        data_path=args.data_path,
        download_dataset=args.download_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.save_checkpoint,
        metrics_path=args.metrics_path,
        grad_clip_norm=args.grad_clip_norm,
    )
    train_model(model_config=model_config, train_config=train_config)


if __name__ == "__main__":
    main()
