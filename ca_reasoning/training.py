from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CharVocab, build_datasets, maybe_download_dataset, read_text
from .models import CellularAutomataLanguageModel, TransformerNextTokenModel


@dataclass(frozen=True)
class ModelConfig:
    model: str
    context_len: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    num_layers: int = 4
    num_heads: int = 4
    ca_steps: int = 8
    rule_sharing: str = "unshared"
    grid_layout: str = "row_major_2d"
    neighborhood: str = "3x3_masked"
    position_mode: str = "both"


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("data/tinyshakespeare.txt")
    download_dataset: bool = False
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 20
    val_fraction: float = 0.1
    seed: int = 1337
    device: str | None = None
    checkpoint_path: Path | None = None
    metrics_path: Path | None = None
    grad_clip_norm: float = 1.0


@dataclass
class TrainingRun:
    model: torch.nn.Module
    vocab: CharVocab
    history: list[dict[str, float | int]]
    data_path: Path
    device: str
    train_windows: int
    val_windows: int
    parameter_count: int
    final_train_loss: float
    final_train_bpc: float
    final_val_loss: float
    final_val_bpc: float
    best_val_loss: float
    best_val_bpc: float
    best_step: int
    metrics: dict[str, Any]


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    vocab: CharVocab
    model_config: ModelConfig
    train_config: TrainConfig
    history: list[dict[str, float | int]]
    metrics: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def maybe_prepare_data(data_path: Path, should_download: bool) -> Path:
    if should_download or not data_path.exists():
        return maybe_download_dataset(data_path)
    return data_path


def build_vocab_and_datasets(
    data_path: Path,
    context_len: int,
    val_fraction: float,
    should_download: bool,
) -> tuple[Path, CharVocab, Any, Any]:
    prepared_path = maybe_prepare_data(data_path, should_download=should_download)
    text = read_text(prepared_path)
    vocab, train_dataset, val_dataset = build_datasets(
        text=text,
        context_len=context_len,
        val_fraction=val_fraction,
    )
    return prepared_path, vocab, train_dataset, val_dataset


def infinite_batches(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def cross_entropy_and_bpc(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss, float(loss.item() / math.log(2.0))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss, _ = cross_entropy_and_bpc(logits, targets)
            losses.append(loss.item())

    if model_was_training:
        model.train()

    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "bpc": mean_loss / math.log(2.0)}


def build_model(model_config: ModelConfig, vocab_size: int) -> torch.nn.Module:
    if model_config.model == "transformer":
        return TransformerNextTokenModel(
            vocab_size=vocab_size,
            context_len=model_config.context_len,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )
    if model_config.model == "ca":
        return CellularAutomataLanguageModel(
            vocab_size=vocab_size,
            context_len=model_config.context_len,
            hidden_dim=model_config.hidden_dim,
            num_steps=model_config.ca_steps,
            dropout=model_config.dropout,
            rule_sharing=model_config.rule_sharing,
            grid_layout=model_config.grid_layout,
            neighborhood=model_config.neighborhood,
            position_mode=model_config.position_mode,
        )
    raise ValueError(f"Unknown model type: {model_config.model}")


def _serialize_train_config(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("data_path", "checkpoint_path", "metrics_path"):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    return payload


def _deserialize_train_config(payload: dict[str, Any]) -> TrainConfig:
    restored = dict(payload)
    for key in ("data_path", "checkpoint_path", "metrics_path"):
        value = restored.get(key)
        if value is not None:
            restored[key] = Path(value)
    return TrainConfig(**restored)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    vocab: CharVocab,
    model_config: ModelConfig,
    train_config: TrainConfig,
    history: list[dict[str, float | int]],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": _serialize_train_config(train_config),
            "vocab": {"stoi": vocab.stoi, "itos": vocab.itos},
            "history": history,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=device)
    vocab_payload = checkpoint["vocab"]
    vocab = CharVocab(stoi=vocab_payload["stoi"], itos=vocab_payload["itos"])
    model_config = ModelConfig(**checkpoint["model_config"])
    train_config = _deserialize_train_config(checkpoint["train_config"])

    model = build_model(model_config=model_config, vocab_size=vocab.size)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        vocab=vocab,
        model_config=model_config,
        train_config=train_config,
        history=checkpoint.get("history", []),
        metrics=checkpoint.get("metrics", {}),
    )


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_training_step(
    model_config: ModelConfig,
    vocab_size: int,
    batch_size: int,
    device_name: str | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 5,
    timed_steps: int = 10,
    seed: int = 1337,
) -> dict[str, float | int | None | str]:
    set_seed(seed)
    device = resolve_device(device_name)
    model = build_model(model_config=model_config, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, model_config.context_len),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, model_config.context_len),
        device=device,
    )

    def run_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss, _ = cross_entropy_and_bpc(logits, targets)
        loss.backward()
        optimizer.step()

    model.train()
    for _ in range(warmup_steps):
        run_step()
    _sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms: list[float] = []
    for _ in range(timed_steps):
        _sync_device(device)
        start = time.perf_counter()
        run_step()
        _sync_device(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    peak_memory_mb: float | None = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    mean_step_ms = statistics.fmean(timings_ms)
    tokens_per_sec = (batch_size * model_config.context_len * 1000.0) / mean_step_ms

    return {
        "device": str(device),
        "parameter_count": parameter_count(model),
        "median_step_ms": float(statistics.median(timings_ms)),
        "mean_step_ms": float(mean_step_ms),
        "tokens_per_sec": float(tokens_per_sec),
        "peak_memory_mb": peak_memory_mb,
    }


def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    print_fn: Callable[[str], None] = print,
) -> TrainingRun:
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)

    data_path, vocab, train_dataset, val_dataset = build_vocab_and_datasets(
        data_path=train_config.data_path,
        context_len=model_config.context_len,
        val_fraction=train_config.val_fraction,
        should_download=train_config.download_dataset,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = build_model(model_config=model_config, vocab_size=vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    batch_stream = infinite_batches(train_loader)
    params = parameter_count(model)

    print_fn(f"device: {device}")
    print_fn(f"dataset: {data_path}")
    print_fn(f"train windows: {len(train_dataset):,}")
    print_fn(f"val windows: {len(val_dataset):,}")
    print_fn(f"vocab size: {vocab.size}")
    print_fn(f"parameters: {params:,}")
    if isinstance(model, CellularAutomataLanguageModel):
        print_fn(
            "ca grid: "
            f"{model.grid_height}x{model.grid_width} "
            f"(capacity={model.grid_capacity}, steps={model.num_steps}, "
            f"sharing={model.rule_sharing}, layout={model.grid_layout}, "
            f"neighborhood={model.neighborhood}, position={model.position_mode})"
        )

    history: list[dict[str, float | int]] = []
    final_train_loss = float("nan")
    final_train_bpc = float("nan")
    final_val_loss = float("nan")
    final_val_bpc = float("nan")
    best_val_loss = float("inf")
    best_val_bpc = float("inf")
    best_step = -1

    model.train()
    for step in range(1, train_config.max_steps + 1):
        inputs, targets = next(batch_stream)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss, train_bpc = cross_entropy_and_bpc(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=train_config.grad_clip_norm,
        )
        optimizer.step()

        if (
            step == 1
            or step % train_config.eval_interval == 0
            or step == train_config.max_steps
        ):
            final_train_loss = float(loss.item())
            final_train_bpc = train_bpc
            eval_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=train_config.eval_batches,
            )
            final_val_loss = float(eval_metrics["loss"])
            final_val_bpc = float(eval_metrics["bpc"])
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_val_bpc = final_val_bpc
                best_step = step

            history.append(
                {
                    "step": step,
                    "train_loss": final_train_loss,
                    "train_bpc": final_train_bpc,
                    "val_loss": final_val_loss,
                    "val_bpc": final_val_bpc,
                }
            )
            print_fn(
                f"step {step:5d} | train_loss {final_train_loss:.4f} "
                f"| train_bpc {final_train_bpc:.4f} | val_loss {final_val_loss:.4f} "
                f"| val_bpc {final_val_bpc:.4f}"
            )

    metrics = {
        "data_path": str(data_path),
        "device": str(device),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "vocab_size": vocab.size,
        "parameter_count": params,
        "final_train_loss": final_train_loss,
        "final_train_bpc": final_train_bpc,
        "final_val_loss": final_val_loss,
        "final_val_bpc": final_val_bpc,
        "best_val_loss": best_val_loss,
        "best_val_bpc": best_val_bpc,
        "best_step": best_step,
    }

    if train_config.metrics_path is not None:
        train_config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_config": asdict(model_config),
            "train_config": _serialize_train_config(train_config),
            "history": history,
            "metrics": metrics,
        }
        train_config.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if train_config.checkpoint_path is not None:
        save_checkpoint(
            path=train_config.checkpoint_path,
            model=model,
            vocab=vocab,
            model_config=model_config,
            train_config=train_config,
            history=history,
            metrics=metrics,
        )

    return TrainingRun(
        model=model,
        vocab=vocab,
        history=history,
        data_path=data_path,
        device=str(device),
        train_windows=len(train_dataset),
        val_windows=len(val_dataset),
        parameter_count=params,
        final_train_loss=final_train_loss,
        final_train_bpc=final_train_bpc,
        final_val_loss=final_val_loss,
        final_val_bpc=final_val_bpc,
        best_val_loss=best_val_loss,
        best_val_bpc=best_val_bpc,
        best_step=best_step,
        metrics=metrics,
    )
