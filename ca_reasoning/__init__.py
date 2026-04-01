"""Small experiments for transformer-vs-cellular language models."""

from .data import (
    CharVocab,
    NextCharDataset,
    build_datasets,
    maybe_download_dataset,
    maybe_download_enwik8,
    maybe_download_text8,
    maybe_download_tinyshakespeare,
)
from .models import CellularAutomataLanguageModel, TransformerNextTokenModel
from .training import ModelConfig, TrainConfig, benchmark_training_step, train_model

__all__ = [
    "CellularAutomataLanguageModel",
    "CharVocab",
    "ModelConfig",
    "NextCharDataset",
    "TrainConfig",
    "TransformerNextTokenModel",
    "benchmark_training_step",
    "build_datasets",
    "maybe_download_dataset",
    "maybe_download_enwik8",
    "maybe_download_text8",
    "maybe_download_tinyshakespeare",
    "train_model",
]
