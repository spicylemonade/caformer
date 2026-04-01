import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ca_reasoning.data import NextCharDataset, build_vocab
from ca_reasoning.training import (
    ModelConfig,
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
)
from ca_reasoning.models import CellularAutomataLanguageModel, TransformerNextTokenModel


class DatasetSmokeTest(unittest.TestCase):
    def test_next_char_dataset_returns_shifted_targets(self) -> None:
        text = "To be, or not to be, that is the question.\n" * 20
        vocab = build_vocab(text)
        dataset = NextCharDataset(vocab.encode(text), context_len=12)

        inputs, targets = dataset[3]
        self.assertEqual(tuple(inputs.shape), (12,))
        self.assertEqual(tuple(targets.shape), (12,))
        self.assertTrue(torch.equal(inputs[1:], targets[:-1]))


class ModelSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.vocab_size = 19
        self.tokens = torch.randint(0, self.vocab_size, (4, 16))

    def test_transformer_forward_shape(self) -> None:
        model = TransformerNextTokenModel(
            vocab_size=self.vocab_size,
            context_len=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        logits = model(self.tokens)
        self.assertEqual(tuple(logits.shape), (4, 16, self.vocab_size))

    def test_ca_forward_shape_and_trace(self) -> None:
        model = CellularAutomataLanguageModel(
            vocab_size=self.vocab_size,
            context_len=16,
            hidden_dim=32,
            num_steps=5,
            dropout=0.0,
            rule_sharing="unshared",
        )
        logits, trace = model(self.tokens, return_trace=True)
        self.assertEqual(tuple(logits.shape), (4, 16, self.vocab_size))
        self.assertEqual(len(trace), 6)

    def test_ca_is_causal_with_future_token_perturbation(self) -> None:
        model = CellularAutomataLanguageModel(
            vocab_size=self.vocab_size,
            context_len=16,
            hidden_dim=32,
            num_steps=3,
            dropout=0.0,
            rule_sharing="unshared",
            grid_layout="row_major_2d",
            neighborhood="3x3_masked",
            position_mode="both",
        )
        model.eval()

        baseline = self.tokens.clone()
        perturbed = baseline.clone()
        perturbed[:, 10:] = torch.randint(0, self.vocab_size, perturbed[:, 10:].shape)

        with torch.no_grad():
            baseline_logits = model(baseline)
            perturbed_logits = model(perturbed)

        self.assertTrue(
            torch.allclose(
                baseline_logits[:, :10, :],
                perturbed_logits[:, :10, :],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_checkpoint_roundtrip_for_ca_model(self) -> None:
        model = CellularAutomataLanguageModel(
            vocab_size=self.vocab_size,
            context_len=16,
            hidden_dim=32,
            num_steps=3,
            dropout=0.0,
            rule_sharing="shared",
            grid_layout="tape_1d",
            neighborhood="5x5_masked",
            position_mode="grid_only",
        )
        vocab = build_vocab("abcdefghijklmnopqrs")
        model_config = ModelConfig(
            model="ca",
            context_len=16,
            hidden_dim=32,
            dropout=0.0,
            ca_steps=3,
            rule_sharing="shared",
            grid_layout="tape_1d",
            neighborhood="5x5_masked",
            position_mode="grid_only",
        )
        train_config = TrainConfig(
            data_path=Path("data/example.txt"),
            max_steps=1,
            checkpoint_path=Path("checkpoints/test.pt"),
        )

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                vocab=vocab,
                model_config=model_config,
                train_config=train_config,
                history=[{"step": 1, "train_loss": 1.23, "val_loss": 1.11}],
                metrics={"parameter_count": 1234},
            )
            loaded = load_checkpoint(checkpoint_path, device="cpu")

        self.assertIsInstance(loaded.model, CellularAutomataLanguageModel)
        self.assertEqual(loaded.model_config.ca_steps, 3)
        self.assertEqual(loaded.model_config.rule_sharing, "shared")
        self.assertEqual(loaded.model_config.grid_layout, "tape_1d")
        self.assertEqual(loaded.model_config.neighborhood, "5x5_masked")
        self.assertEqual(loaded.model_config.position_mode, "grid_only")
        self.assertEqual(loaded.vocab.size, vocab.size)
        tokens = self.tokens % loaded.vocab.size
        logits = loaded.model(tokens)
        self.assertEqual(tuple(logits.shape), (4, 16, loaded.vocab.size))


if __name__ == "__main__":
    unittest.main()
