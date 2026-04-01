from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
TEXT8_URL = "https://mattmahoney.net/dc/text8.zip"
ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"


@dataclass(frozen=True)
class CharVocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


class NextCharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, encoded_text: list[int], context_len: int) -> None:
        if len(encoded_text) <= context_len:
            raise ValueError(
                f"Need more than {context_len} encoded tokens, got {len(encoded_text)}."
            )

        self.data = torch.tensor(encoded_text, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return self.data.size(0) - self.context_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = index + self.context_len
        return self.data[start:end], self.data[start + 1 : end + 1]


def build_vocab(text: str) -> CharVocab:
    symbols = sorted(set(text))
    return CharVocab(stoi={ch: i for i, ch in enumerate(symbols)}, itos=symbols)


def _dataset_kind(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "tinyshakespeare" in name:
        return "tinyshakespeare"
    if "text8" in name:
        return "text8"
    if "enwik8" in name:
        return "enwik8"
    return "custom"


def _download_zip_member(
    destination: str | Path,
    url: str,
    member_name: str,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return destination

    archive_path = destination.parent / f"{member_name}.zip"
    urlretrieve(url, archive_path)
    with ZipFile(archive_path) as archive:
        destination.write_bytes(archive.read(member_name))
    archive_path.unlink(missing_ok=True)
    return destination


def maybe_download_tinyshakespeare(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        urlretrieve(TINY_SHAKESPEARE_URL, destination)

    return destination


def maybe_download_text8(destination: str | Path) -> Path:
    return _download_zip_member(destination=destination, url=TEXT8_URL, member_name="text8")


def maybe_download_enwik8(destination: str | Path) -> Path:
    return _download_zip_member(destination=destination, url=ENWIK8_URL, member_name="enwik8")


def maybe_download_dataset(destination: str | Path) -> Path:
    dataset_kind = _dataset_kind(destination)
    if dataset_kind == "tinyshakespeare":
        return maybe_download_tinyshakespeare(destination)
    if dataset_kind == "text8":
        return maybe_download_text8(destination)
    if dataset_kind == "enwik8":
        return maybe_download_enwik8(destination)

    destination = Path(destination)
    if destination.exists():
        return destination
    raise ValueError(
        "Cannot infer a known dataset download for "
        f"{destination}. Use a path containing tinyshakespeare, text8, or enwik8."
    )


def read_text(path: str | Path) -> str:
    path = Path(path)
    if _dataset_kind(path) == "enwik8":
        return path.read_bytes().decode("latin-1")
    return path.read_text(encoding="utf-8")


def build_datasets(
    text: str,
    context_len: int,
    val_fraction: float = 0.1,
) -> tuple[CharVocab, NextCharDataset, NextCharDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}.")

    split_index = int(len(text) * (1.0 - val_fraction))
    minimum_tokens = context_len + 1
    if split_index < minimum_tokens or len(text) - split_index < minimum_tokens:
        raise ValueError(
            "Text is too short for the requested context length and validation split."
        )

    vocab = build_vocab(text)
    train_text = text[:split_index]
    val_text = text[split_index:]

    train_dataset = NextCharDataset(vocab.encode(train_text), context_len=context_len)
    val_dataset = NextCharDataset(vocab.encode(val_text), context_len=context_len)
    return vocab, train_dataset, val_dataset
