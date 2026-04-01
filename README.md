# CA Language Model

This repo studies one question:

Can a causally masked neural cellular automaton replace transformer attention for language modeling?

The current codebase is no longer a toy last-token probe. It now supports:

- Full causal next-token prediction over every position in the window
- A transformer baseline
- A strictly causal CA language model
- Shared-rule and unshared-rule CA variants
- `row_major_2d` and `tape_1d` layouts
- Masked `3x3`, masked `5x5`, and dilated CA neighborhoods
- Parameter-matched and step-time-matched comparisons
- Runtime and memory benchmarking
- Sweep orchestration and JSON aggregation
- Tiny Shakespeare, `text8`, and optional `enwik8`
- Synthetic long-range corpora for copy, delayed copy, brackets, and induction-style recall
- CA trace visualization as animated GIFs

## Architecture

### Transformer baseline

- Token embedding + 1D position embedding
- Causal self-attention blocks
- Feed-forward layers
- Per-position vocabulary head

### CA model

- Token embedding
- Optional 1D sequence positions
- Pack to a 2D grid or 1D tape
- Optional grid position embedding
- Repeated local update rules
- Per-position vocabulary head after unpacking back to sequence order

The CA rule itself is learned by backprop. In the unshared setting, each CA step has its own learned local rule. In the shared setting, all steps reuse one rule.

The causal mask is important: for the default `row_major_2d` layout, each cell can only read row-major-past neighbors and itself, which avoids future-token leakage.

## Install

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Datasets

The training stack will auto-download based on filename:

- `data/tinyshakespeare.txt`
- `data/text8.txt`
- `data/enwik8.bin`

Examples:

```bash
python3 -m ca_reasoning.train --model transformer --data-path data/tinyshakespeare.txt --download-dataset
python3 -m ca_reasoning.train --model ca --data-path data/text8.txt --download-dataset
python3 -m ca_reasoning.compare --data-path data/enwik8.bin --download-dataset --max-steps 1
```

## Train A Model

```bash
python3 -m ca_reasoning.train \
  --model ca \
  --data-path data/tinyshakespeare.txt \
  --download-dataset \
  --context-len 128 \
  --hidden-dim 104 \
  --ca-steps 8 \
  --rule-sharing shared \
  --grid-layout row_major_2d \
  --ca-neighborhood 3x3_masked \
  --position-mode both \
  --save-checkpoint checkpoints/ca.pt \
  --metrics-path results/ca_metrics.json
```

## Compare Models

Parameter-matched comparison:

```bash
python3 -m ca_reasoning.compare \
  --data-path data/tinyshakespeare.txt \
  --download-dataset \
  --matching-mode parameter \
  --transformer-hidden-dim 96 \
  --transformer-num-layers 4 \
  --transformer-num-heads 4 \
  --ca-steps 8 \
  --ca-rule-sharing shared \
  --hidden-dim-min 16 \
  --hidden-dim-max 160 \
  --hidden-dim-step 1 \
  --max-steps 150 \
  --tag demo_parameter
```

Step-time-matched comparison:

```bash
python3 -m ca_reasoning.compare \
  --data-path data/text8.txt \
  --download-dataset \
  --matching-mode step_time \
  --transformer-hidden-dim 96 \
  --transformer-num-layers 4 \
  --transformer-num-heads 4 \
  --ca-steps 8 \
  --ca-rule-sharing unshared \
  --hidden-dim-min 16 \
  --hidden-dim-max 160 \
  --hidden-dim-step 8 \
  --benchmark-warmup-steps 2 \
  --benchmark-timed-steps 4 \
  --max-steps 150 \
  --tag demo_step_time
```

## Benchmark Scaling

```bash
python3 -m ca_reasoning.benchmark \
  --data-path data/tinyshakespeare.txt \
  --download-dataset \
  --contexts 128 256 512 1024 \
  --batch-size 32 \
  --transformer-hidden-dim 96 \
  --transformer-num-layers 4 \
  --transformer-num-heads 4 \
  --ca-hidden-dim 104 \
  --ca-steps 8 \
  --ca-rule-sharing shared \
  --results-path results/benchmarks_parameter_shared.json
```

## Run Preset Sweeps

Phase 1 core studies:

```bash
python3 -m ca_reasoning.sweep \
  --preset phase1_core \
  --datasets tinyshakespeare text8 \
  --seeds 7 42 1337 \
  --matching-modes parameter step_time \
  --context-len 128 \
  --batch-size 64 \
  --max-steps 150
```

Phase 1 ablations:

```bash
python3 -m ca_reasoning.sweep \
  --preset phase1_ablations \
  --datasets tinyshakespeare \
  --seeds 7 42 1337 \
  --ca-steps-sweep 4 8 12 \
  --position-modes seq_only grid_only both \
  --context-len 128 \
  --batch-size 64 \
  --max-steps 150
```

Phase 2 extension pilots:

```bash
python3 -m ca_reasoning.sweep \
  --preset phase2_extensions \
  --datasets tinyshakespeare \
  --seeds 7 \
  --context-len 128 \
  --batch-size 64 \
  --max-steps 150
```

Aggregate any results directory:

```bash
python3 -m ca_reasoning.aggregate --results-dir results/studies --output-path results/studies_summary.json
```

## Synthetic Long-Range Corpora

Generate one:

```bash
python3 -m ca_reasoning.synthetic \
  --task delayed_copy \
  --output-path data/synthetic_delayed_copy.txt \
  --num-examples 40000 \
  --min-length 8 \
  --max-length 16 \
  --delay-length 48
```

Supported tasks:

- `copy`
- `delayed_copy`
- `brackets`
- `induction`

## Visualize The CA

After training a CA checkpoint, render its internal evolution:

```bash
python3 -m ca_reasoning.visualize \
  --checkpoint checkpoints/ca.pt \
  --prompt-index 0 \
  --output artifacts/ca_trace.gif
```

Or use a custom prompt:

```bash
python3 -m ca_reasoning.visualize \
  --checkpoint checkpoints/ca.pt \
  --prompt "To be, or not to be"
```

## Results In This Repo

The current run artifacts are already saved:

- `results/studies/phase1_core/summary.json`
- `results/studies_step_time/phase1_core/summary.json`
- `results/studies_ablations/phase1_ablations/summary.json`
- `results/studies_phase2/phase2_extensions/summary.json`
- `results/benchmarks_parameter_shared.json`
- `results/benchmarks_step_time_unshared.json`
- `results/synthetic_copy.json`
- `results/synthetic_delayed_copy.json`
- `results/synthetic_brackets.json`
- `results/synthetic_induction.json`
- `results/enwik8_smoke.json`

## Current Takeaways

- Parameter-matched **shared** 2D CA is substantially better than the transformer baseline after 150 steps on both Tiny Shakespeare and `text8`.
- Parameter-matched **unshared** 2D CA underperforms badly because matching total parameters forces the hidden width to become too small.
- Step-time-matched **unshared** CA is competitive on Tiny Shakespeare and better than the transformer on `text8`, but it needs many more parameters to hit the same step time.
- In the extension sweep, `tape_1d` beats `row_major_2d` for the unshared CA, while `5x5` and dilated neighborhoods were not helpful in the tested setup.
- On the synthetic long-range corpora, the current unshared row-major 2D CA trails the transformer on all four tasks.

These results make the repo useful as an actual research harness: the CA idea clearly has real signal, but the exact rule-sharing and layout choices matter a lot.
