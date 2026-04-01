from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .aggregate import aggregate_results
from .compare import count_parameters_for_config, match_hidden_dim
from .training import (
    ModelConfig,
    TrainConfig,
    benchmark_training_step,
    build_vocab_and_datasets,
    train_model,
)


DEFAULT_DATASETS = {
    "tinyshakespeare": Path("data/tinyshakespeare.txt"),
    "text8": Path("data/text8.txt"),
    "enwik8": Path("data/enwik8.bin"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preset sweeps for the CA-vs-transformer study."
    )
    parser.add_argument(
        "--preset",
        choices=("phase1_core", "phase1_ablations", "phase2_extensions"),
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 1337])
    parser.add_argument("--matching-modes", nargs="+", default=None)

    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--transformer-hidden-dim", type=int, default=96)
    parser.add_argument("--transformer-num-layers", type=int, default=4)
    parser.add_argument("--transformer-num-heads", type=int, default=4)

    parser.add_argument("--ca-hidden-dim", type=int, default=None)
    parser.add_argument("--ca-steps", type=int, default=8)
    parser.add_argument("--ca-steps-sweep", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--position-modes", nargs="+", default=["seq_only", "grid_only", "both"])
    parser.add_argument("--hidden-dim-min", type=int, default=16)
    parser.add_argument("--hidden-dim-max", type=int, default=256)
    parser.add_argument("--hidden-dim-step", type=int, default=4)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=3)
    parser.add_argument("--benchmark-timed-steps", type=int, default=6)

    parser.add_argument("--results-dir", type=Path, default=Path("results/sweeps"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/sweeps"))
    return parser.parse_args()


def resolve_dataset_path(dataset: str) -> Path:
    return DEFAULT_DATASETS.get(dataset, Path(dataset))


def benchmark_kwargs(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "device_name": args.device,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.benchmark_warmup_steps,
        "timed_steps": args.benchmark_timed_steps,
        "seed": seed,
    }


def build_train_config(
    args: argparse.Namespace,
    data_path: Path,
    seed: int,
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
        seed=seed,
        device=args.device,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        grad_clip_norm=args.grad_clip_norm,
    )


def run_compare_case(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    seed: int,
    match_mode: str,
    result_path: Path,
    checkpoint_root: Path,
    ca_rule_sharing: str,
    ca_steps: int,
    ca_position_mode: str,
    ca_grid_layout: str = "row_major_2d",
    ca_neighborhood: str = "3x3_masked",
    tag: str,
    experiment_group: str,
) -> dict[str, Any]:
    data_path, vocab, _, _ = build_vocab_and_datasets(
        data_path=dataset_path,
        context_len=args.context_len,
        val_fraction=args.val_fraction,
        should_download=args.download_datasets,
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

    transformer_benchmark = None
    target_value = float(transformer_param_count)
    if match_mode == "step_time":
        transformer_benchmark = benchmark_training_step(
            model_config=transformer_config,
            vocab_size=vocab.size,
            **benchmark_kwargs(args, seed),
        )
        target_value = float(transformer_benchmark["mean_step_ms"])

    ca_template = ModelConfig(
        model="ca",
        context_len=args.context_len,
        hidden_dim=args.ca_hidden_dim or args.transformer_hidden_dim,
        dropout=args.dropout,
        num_heads=args.transformer_num_heads,
        ca_steps=ca_steps,
        rule_sharing=ca_rule_sharing,
        grid_layout=ca_grid_layout,
        neighborhood=ca_neighborhood,
        position_mode=ca_position_mode,
    )
    if args.ca_hidden_dim is None:
        ca_config, ca_match_details = match_hidden_dim(
            base_config=ca_template,
            vocab_size=vocab.size,
            target_value=target_value,
            minimum=args.hidden_dim_min,
            maximum=args.hidden_dim_max,
            hidden_dim_step=args.hidden_dim_step,
            match_mode=match_mode,
            benchmark_kwargs=benchmark_kwargs(args, seed) if match_mode == "step_time" else None,
        )
    else:
        ca_config = ca_template
        ca_match_details = (
            benchmark_training_step(
                model_config=ca_config,
                vocab_size=vocab.size,
                **benchmark_kwargs(args, seed),
            )
            if match_mode == "step_time"
            else {"parameter_count": count_parameters_for_config(ca_config, vocab_size=vocab.size)}
        )

    transformer_train_config = build_train_config(
        args=args,
        data_path=data_path,
        seed=seed,
        checkpoint_path=checkpoint_root / f"{tag}_transformer.pt",
        metrics_path=checkpoint_root / f"{tag}_transformer_metrics.json",
    )
    ca_train_config = build_train_config(
        args=args,
        data_path=data_path,
        seed=seed,
        checkpoint_path=checkpoint_root / f"{tag}_ca.pt",
        metrics_path=checkpoint_root / f"{tag}_ca_metrics.json",
    )

    print(f"\n== running {tag} ==")
    transformer_run = train_model(
        model_config=transformer_config,
        train_config=transformer_train_config,
    )
    ca_run = train_model(
        model_config=ca_config,
        train_config=ca_train_config,
    )

    result = {
        "kind": "compare",
        "tag": tag,
        "experiment_group": experiment_group,
        "seed": seed,
        "match_mode": match_mode,
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
            "benchmark": ca_match_details if match_mode == "step_time" else None,
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
    if match_mode == "step_time" and transformer_benchmark is not None:
        result["delta"]["step_time_gap_ms"] = (
            float(ca_match_details["mean_step_ms"]) - float(transformer_benchmark["mean_step_ms"])
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def phase1_core_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = args.datasets or ["tinyshakespeare", "text8"]
    matching_modes = args.matching_modes or ["parameter", "step_time"]
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in args.seeds:
            for match_mode in matching_modes:
                for rule_sharing in ("shared", "unshared"):
                    jobs.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "match_mode": match_mode,
                            "ca_rule_sharing": rule_sharing,
                            "ca_steps": args.ca_steps,
                            "ca_position_mode": "both",
                            "ca_grid_layout": "row_major_2d",
                            "ca_neighborhood": "3x3_masked",
                            "group": "phase1_core",
                        }
                    )
    return jobs


def phase1_ablation_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = args.datasets or ["tinyshakespeare"]
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in args.seeds:
            for ca_steps in args.ca_steps_sweep:
                jobs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "match_mode": "parameter",
                        "ca_rule_sharing": "unshared",
                        "ca_steps": ca_steps,
                        "ca_position_mode": "both",
                        "ca_grid_layout": "row_major_2d",
                        "ca_neighborhood": "3x3_masked",
                        "group": "ablation_ca_steps",
                    }
                )
            for position_mode in args.position_modes:
                jobs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "match_mode": "parameter",
                        "ca_rule_sharing": "unshared",
                        "ca_steps": args.ca_steps,
                        "ca_position_mode": position_mode,
                        "ca_grid_layout": "row_major_2d",
                        "ca_neighborhood": "3x3_masked",
                        "group": "ablation_position_mode",
                    }
                )
            for rule_sharing in ("shared", "unshared"):
                jobs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "match_mode": "parameter",
                        "ca_rule_sharing": rule_sharing,
                        "ca_steps": args.ca_steps,
                        "ca_position_mode": "both",
                        "ca_grid_layout": "row_major_2d",
                        "ca_neighborhood": "3x3_masked",
                        "group": "ablation_rule_sharing",
                    }
                )
    return jobs


def phase2_extension_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = args.datasets or ["tinyshakespeare"]
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        seed = args.seeds[0]
        for grid_layout in ("row_major_2d", "tape_1d"):
            jobs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "match_mode": "parameter",
                    "ca_rule_sharing": "unshared",
                    "ca_steps": args.ca_steps,
                    "ca_position_mode": "both",
                    "ca_grid_layout": grid_layout,
                    "ca_neighborhood": "3x3_masked",
                    "group": "extension_grid_layout",
                }
            )
        for neighborhood in ("3x3_masked", "5x5_masked", "3x3_dilated"):
            jobs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "match_mode": "parameter",
                    "ca_rule_sharing": "unshared",
                    "ca_steps": args.ca_steps,
                    "ca_position_mode": "both",
                    "ca_grid_layout": "row_major_2d",
                    "ca_neighborhood": neighborhood,
                    "group": "extension_neighborhood",
                }
            )
    return jobs


def main() -> None:
    args = parse_args()
    if args.preset == "phase1_core":
        jobs = phase1_core_jobs(args)
    elif args.preset == "phase1_ablations":
        jobs = phase1_ablation_jobs(args)
    else:
        jobs = phase2_extension_jobs(args)

    results_root = args.results_dir / args.preset
    checkpoints_root = args.checkpoint_dir / args.preset
    completed_results: list[str] = []

    for job in jobs:
        dataset_name = Path(job["dataset"]).stem if job["dataset"] not in DEFAULT_DATASETS else job["dataset"]
        tag = (
            f"{job['group']}_{dataset_name}_seed{job['seed']}_"
            f"{job['match_mode']}_{job['ca_rule_sharing']}_steps{job['ca_steps']}_"
            f"pos{job['ca_position_mode']}_{job['ca_grid_layout']}_{job['ca_neighborhood']}"
        )
        result_path = results_root / f"{tag}.json"
        checkpoint_root = checkpoints_root / tag
        run_compare_case(
            args=args,
            dataset_path=resolve_dataset_path(job["dataset"]),
            seed=job["seed"],
            match_mode=job["match_mode"],
            result_path=result_path,
            checkpoint_root=checkpoint_root,
            ca_rule_sharing=job["ca_rule_sharing"],
            ca_steps=job["ca_steps"],
            ca_position_mode=job["ca_position_mode"],
            ca_grid_layout=job["ca_grid_layout"],
            ca_neighborhood=job["ca_neighborhood"],
            tag=tag,
            experiment_group=job["group"],
        )
        completed_results.append(str(result_path))

    manifest_path = results_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "sweep_manifest",
                "preset": args.preset,
                "results": completed_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = results_root / "summary.json"
    summary = aggregate_results(results_root)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved sweep manifest to: {manifest_path}")
    print(f"saved sweep summary to: {summary_path}")


if __name__ == "__main__":
    main()
