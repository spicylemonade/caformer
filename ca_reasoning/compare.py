from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .training import (
    ModelConfig,
    TrainConfig,
    benchmark_training_step,
    build_model,
    build_vocab_and_datasets,
    parameter_count,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched-budget transformer vs CA comparison."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--matching-mode",
        choices=("parameter", "step_time"),
        default="parameter",
    )
    parser.add_argument("--hidden-dim-step", type=int, default=4)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=3)
    parser.add_argument("--benchmark-timed-steps", type=int, default=6)

    parser.add_argument("--transformer-hidden-dim", type=int, default=96)
    parser.add_argument("--transformer-num-layers", type=int, default=4)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--ca-steps", type=int, default=8)
    parser.add_argument("--ca-hidden-dim", type=int, default=None)
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
    parser.add_argument("--hidden-dim-min", type=int, default=16)
    parser.add_argument("--hidden-dim-max", type=int, default=256)

    parser.add_argument("--tag", type=str, default="pilot")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/head_to_head"),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/head_to_head.json"),
    )
    return parser.parse_args()


def count_parameters_for_config(model_config: ModelConfig, vocab_size: int) -> int:
    model = build_model(model_config=model_config, vocab_size=vocab_size)
    return parameter_count(model)


def candidate_hidden_dims(
    model_name: str,
    minimum: int,
    maximum: int,
    num_heads: int,
    hidden_dim_step: int,
) -> list[int]:
    if model_name == "transformer":
        return [
            hidden_dim
            for hidden_dim in range(minimum, maximum + 1, max(hidden_dim_step, 1))
            if hidden_dim % num_heads == 0
        ]
    return list(range(minimum, maximum + 1, max(hidden_dim_step, 1)))


def match_hidden_dim(
    base_config: ModelConfig,
    vocab_size: int,
    target_value: float,
    minimum: int,
    maximum: int,
    hidden_dim_step: int,
    match_mode: str,
    benchmark_kwargs: dict[str, Any] | None = None,
) -> tuple[ModelConfig, dict[str, Any]]:
    best_config: ModelConfig | None = None
    best_gap: float | None = None
    best_details: dict[str, Any] = {}

    for hidden_dim in candidate_hidden_dims(
        model_name=base_config.model,
        minimum=minimum,
        maximum=maximum,
        num_heads=base_config.num_heads,
        hidden_dim_step=hidden_dim_step,
    ):
        candidate = replace(base_config, hidden_dim=hidden_dim)
        if match_mode == "parameter":
            value = float(count_parameters_for_config(candidate, vocab_size=vocab_size))
            details = {"parameter_count": int(value)}
        else:
            if benchmark_kwargs is None:
                raise ValueError("benchmark_kwargs are required for step-time matching.")
            details = benchmark_training_step(
                model_config=candidate,
                vocab_size=vocab_size,
                **benchmark_kwargs,
            )
            value = float(details["mean_step_ms"])

        gap = abs(value - target_value)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_config = candidate
            best_details = details

    if best_config is None:
        raise ValueError("No valid hidden dimension candidates were found.")

    return best_config, best_details


def build_train_config(
    args: argparse.Namespace,
    data_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
) -> TrainConfig:
    return TrainConfig(
        data_path=data_path,
        download_dataset=False,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


def main() -> None:
    args = parse_args()
    data_path, vocab, _, _ = build_vocab_and_datasets(
        data_path=args.data_path,
        context_len=args.context_len,
        val_fraction=args.val_fraction,
        should_download=args.download_dataset,
    )

    transformer_config = ModelConfig(
        model="transformer",
        context_len=args.context_len,
        hidden_dim=args.transformer_hidden_dim,
        dropout=args.dropout,
        num_layers=args.transformer_num_layers,
        num_heads=args.transformer_num_heads,
    )
    transformer_param_count = count_parameters_for_config(
        transformer_config,
        vocab_size=vocab.size,
    )

    benchmark_kwargs = {
        "batch_size": args.batch_size,
        "device_name": args.device,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.benchmark_warmup_steps,
        "timed_steps": args.benchmark_timed_steps,
        "seed": args.seed,
    }
    transformer_benchmark: dict[str, Any] | None = None
    target_value = float(transformer_param_count)
    if args.matching_mode == "step_time":
        transformer_benchmark = benchmark_training_step(
            model_config=transformer_config,
            vocab_size=vocab.size,
            **benchmark_kwargs,
        )
        target_value = float(transformer_benchmark["mean_step_ms"])

    ca_template = ModelConfig(
        model="ca",
        context_len=args.context_len,
        hidden_dim=args.ca_hidden_dim or args.transformer_hidden_dim,
        dropout=args.dropout,
        num_heads=args.transformer_num_heads,
        ca_steps=args.ca_steps,
        rule_sharing=args.ca_rule_sharing,
        grid_layout=args.ca_grid_layout,
        neighborhood=args.ca_neighborhood,
        position_mode=args.ca_position_mode,
    )
    if args.ca_hidden_dim is None:
        ca_config, ca_match_details = match_hidden_dim(
            base_config=ca_template,
            vocab_size=vocab.size,
            target_value=target_value,
            minimum=args.hidden_dim_min,
            maximum=args.hidden_dim_max,
            hidden_dim_step=args.hidden_dim_step,
            match_mode=args.matching_mode,
            benchmark_kwargs=benchmark_kwargs if args.matching_mode == "step_time" else None,
        )
    else:
        ca_config = ca_template
        ca_match_details = (
            benchmark_training_step(
                model_config=ca_config,
                vocab_size=vocab.size,
                **benchmark_kwargs,
            )
            if args.matching_mode == "step_time"
            else {"parameter_count": count_parameters_for_config(ca_config, vocab_size=vocab.size)}
        )

    ca_param_count = count_parameters_for_config(ca_config, vocab_size=vocab.size)

    print("matched setup")
    print(
        f"transformer hidden_dim={transformer_config.hidden_dim}, "
        f"params={transformer_param_count:,}"
    )
    print(
        f"ca hidden_dim={ca_config.hidden_dim}, params={ca_param_count:,}, "
        f"match_mode={args.matching_mode}"
    )
    if transformer_benchmark is not None:
        print(
            "transformer step time: "
            f"{float(transformer_benchmark['mean_step_ms']):.3f} ms"
        )
        print(f"ca step time: {float(ca_match_details['mean_step_ms']):.3f} ms")

    checkpoint_dir = args.checkpoint_dir
    transformer_train_config = build_train_config(
        args=args,
        data_path=data_path,
        checkpoint_path=checkpoint_dir / f"{args.tag}_transformer.pt",
        metrics_path=checkpoint_dir / f"{args.tag}_transformer_metrics.json",
    )
    ca_train_config = build_train_config(
        args=args,
        data_path=data_path,
        checkpoint_path=checkpoint_dir / f"{args.tag}_ca.pt",
        metrics_path=checkpoint_dir / f"{args.tag}_ca_metrics.json",
    )

    print("\n== transformer ==")
    transformer_run = train_model(
        model_config=transformer_config,
        train_config=transformer_train_config,
    )
    print("\n== ca ==")
    ca_run = train_model(
        model_config=ca_config,
        train_config=ca_train_config,
    )

    results = {
        "kind": "compare",
        "tag": args.tag,
        "seed": args.seed,
        "match_mode": args.matching_mode,
        "data_path": str(data_path),
        "dataset_name": Path(data_path).name,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "transformer": {
            "model_config": asdict(transformer_config),
            "parameter_count": transformer_run.parameter_count,
            "final_train_loss": transformer_run.final_train_loss,
            "final_train_bpc": transformer_run.final_train_bpc,
            "final_val_loss": transformer_run.final_val_loss,
            "final_val_bpc": transformer_run.final_val_bpc,
            "best_val_loss": transformer_run.best_val_loss,
            "best_val_bpc": transformer_run.best_val_bpc,
            "best_step": transformer_run.best_step,
            "history": transformer_run.history,
            "checkpoint_path": str(transformer_train_config.checkpoint_path),
            "benchmark": transformer_benchmark,
        },
        "ca": {
            "model_config": asdict(ca_config),
            "parameter_count": ca_run.parameter_count,
            "final_train_loss": ca_run.final_train_loss,
            "final_train_bpc": ca_run.final_train_bpc,
            "final_val_loss": ca_run.final_val_loss,
            "final_val_bpc": ca_run.final_val_bpc,
            "best_val_loss": ca_run.best_val_loss,
            "best_val_bpc": ca_run.best_val_bpc,
            "best_step": ca_run.best_step,
            "history": ca_run.history,
            "checkpoint_path": str(ca_train_config.checkpoint_path),
            "benchmark": ca_match_details if args.matching_mode == "step_time" else None,
        },
        "delta": {
            "val_loss_ca_minus_transformer": (
                ca_run.final_val_loss - transformer_run.final_val_loss
            ),
            "val_bpc_ca_minus_transformer": (
                ca_run.final_val_bpc - transformer_run.final_val_bpc
            ),
            "parameter_gap": ca_run.parameter_count - transformer_run.parameter_count,
        },
    }
    if args.matching_mode == "step_time" and transformer_benchmark is not None:
        results["delta"]["step_time_gap_ms"] = (
            float(ca_match_details["mean_step_ms"]) - float(transformer_benchmark["mean_step_ms"])
        )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n== summary ==")
    print(f"results saved to: {args.results_path}")
    print(f"transformer final val_loss: {transformer_run.final_val_loss:.4f}")
    print(f"ca final val_loss: {ca_run.final_val_loss:.4f}")
    print(
        "val loss delta (ca - transformer): "
        f"{results['delta']['val_loss_ca_minus_transformer']:.4f}"
    )


if __name__ == "__main__":
    main()
