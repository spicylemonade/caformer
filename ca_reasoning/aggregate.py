from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate compare and benchmark JSON results into grouped summaries."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-path", type=Path, default=Path("results/summary.json"))
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    compare_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    benchmark_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for result_path in sorted(results_dir.rglob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["result_path"] = str(result_path)
        kind = payload.get("kind")

        if kind == "compare":
            ca_config = payload["ca"]["model_config"]
            transformer_config = payload["transformer"]["model_config"]
            key = (
                payload.get("experiment_group") or "ungrouped",
                Path(payload["data_path"]).name,
                payload["match_mode"],
                ca_config["rule_sharing"],
                ca_config["position_mode"],
                ca_config["ca_steps"],
                ca_config["grid_layout"],
                ca_config["neighborhood"],
                transformer_config["hidden_dim"],
                transformer_config["num_layers"],
                transformer_config["num_heads"],
                payload.get("batch_size"),
                payload.get("max_steps"),
            )
            compare_groups.setdefault(key, []).append(payload)
        elif kind == "benchmark":
            for benchmark in payload["benchmarks"]:
                model_config = benchmark["model_config"]
                key = (
                    Path(payload["data_path"]).name,
                    benchmark["model"],
                    benchmark["context_len"],
                    model_config["hidden_dim"],
                    model_config.get("ca_steps"),
                    model_config.get("rule_sharing"),
                    model_config.get("grid_layout"),
                    model_config.get("neighborhood"),
                    model_config.get("position_mode"),
                )
                benchmark_groups.setdefault(key, []).append(
                    {
                        "data_path": payload["data_path"],
                        "benchmark": benchmark["benchmark"],
                    }
                )

    compare_summaries: list[dict[str, Any]] = []
    for key, payloads in sorted(compare_groups.items()):
        (
            experiment_group,
            dataset_name,
            match_mode,
            rule_sharing,
            position_mode,
            ca_steps,
            grid_layout,
            neighborhood,
            transformer_hidden_dim,
            transformer_num_layers,
            transformer_num_heads,
            batch_size,
            max_steps,
        ) = key
        compare_summaries.append(
            {
                "experiment_group": experiment_group,
                "dataset_name": dataset_name,
                "match_mode": match_mode,
                "rule_sharing": rule_sharing,
                "position_mode": position_mode,
                "ca_steps": ca_steps,
                "grid_layout": grid_layout,
                "neighborhood": neighborhood,
                "transformer_hidden_dim": transformer_hidden_dim,
                "transformer_num_layers": transformer_num_layers,
                "transformer_num_heads": transformer_num_heads,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "num_runs": len(payloads),
                "seeds": [payload["seed"] for payload in payloads],
                "ca_final_val_bpc_mean": _mean(
                    [payload["ca"]["final_val_bpc"] for payload in payloads]
                ),
                "ca_final_val_bpc_std": _std(
                    [payload["ca"]["final_val_bpc"] for payload in payloads]
                ),
                "transformer_final_val_bpc_mean": _mean(
                    [payload["transformer"]["final_val_bpc"] for payload in payloads]
                ),
                "transformer_final_val_bpc_std": _std(
                    [payload["transformer"]["final_val_bpc"] for payload in payloads]
                ),
                "delta_val_bpc_mean": _mean(
                    [payload["delta"]["val_bpc_ca_minus_transformer"] for payload in payloads]
                ),
                "delta_val_bpc_std": _std(
                    [payload["delta"]["val_bpc_ca_minus_transformer"] for payload in payloads]
                ),
                "result_paths": [payload["result_path"] for payload in payloads],
            }
        )

    benchmark_summaries: list[dict[str, Any]] = []
    for key, payloads in sorted(benchmark_groups.items()):
        (
            dataset_name,
            model_name,
            context_len,
            hidden_dim,
            ca_steps,
            rule_sharing,
            grid_layout,
            neighborhood,
            position_mode,
        ) = key
        benchmark_summaries.append(
            {
                "dataset_name": dataset_name,
                "model": model_name,
                "context_len": context_len,
                "hidden_dim": hidden_dim,
                "ca_steps": ca_steps,
                "rule_sharing": rule_sharing,
                "grid_layout": grid_layout,
                "neighborhood": neighborhood,
                "position_mode": position_mode,
                "num_runs": len(payloads),
                "mean_step_ms_mean": _mean(
                    [payload["benchmark"]["mean_step_ms"] for payload in payloads]
                ),
                "mean_step_ms_std": _std(
                    [payload["benchmark"]["mean_step_ms"] for payload in payloads]
                ),
                "tokens_per_sec_mean": _mean(
                    [payload["benchmark"]["tokens_per_sec"] for payload in payloads]
                ),
                "tokens_per_sec_std": _std(
                    [payload["benchmark"]["tokens_per_sec"] for payload in payloads]
                ),
                "peak_memory_mb_mean": _mean(
                    [
                        float(payload["benchmark"]["peak_memory_mb"] or 0.0)
                        for payload in payloads
                    ]
                ),
                "peak_memory_mb_std": _std(
                    [
                        float(payload["benchmark"]["peak_memory_mb"] or 0.0)
                        for payload in payloads
                    ]
                ),
            }
        )

    return {
        "kind": "aggregate_summary",
        "results_dir": str(results_dir),
        "compare_groups": compare_summaries,
        "benchmark_groups": benchmark_summaries,
    }


def main() -> None:
    args = parse_args()
    summary = aggregate_results(args.results_dir)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved aggregate summary to: {args.output_path}")


if __name__ == "__main__":
    main()
