from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .data import build_datasets, read_text
from .models import CellularAutomataLanguageModel
from .training import LoadedCheckpoint, load_checkpoint, maybe_prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an animated GIF of CA grid evolution."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ca_trace.gif"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cell-size", type=int, default=42)
    parser.add_argument("--frame-duration-ms", type=int, default=300)
    return parser.parse_args()


def display_char(ch: str) -> str:
    if ch == " ":
        return "."
    if ch == "\n":
        return "/"
    if ch == "\t":
        return "t"
    return ch


def display_text(text: str, limit: int = 72) -> str:
    escaped = text.replace("\n", "\\n")
    if len(escaped) <= limit:
        return escaped
    return "..." + escaped[-(limit - 3) :]


def color_map(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    anchors = [
        (0.0, (245, 247, 250)),
        (0.3, (191, 219, 254)),
        (0.6, (96, 165, 250)),
        (1.0, (37, 99, 235)),
    ]
    for (left_pos, left_color), (right_pos, right_color) in zip(anchors, anchors[1:]):
        if value <= right_pos:
            span = (value - left_pos) / max(right_pos - left_pos, 1e-8)
            return tuple(
                int(round(left_channel + span * (right_channel - left_channel)))
                for left_channel, right_channel in zip(left_color, right_color)
            )
    return anchors[-1][1]


def resolve_example(
    loaded: LoadedCheckpoint,
    prompt: str | None,
    prompt_index: int,
    data_path: Path | None,
    should_download: bool,
) -> tuple[torch.Tensor, str, str | None]:
    if prompt is not None:
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        prompt_tail = prompt[-loaded.model_config.context_len :]
        missing = sorted({ch for ch in prompt_tail if ch not in loaded.vocab.stoi})
        if missing:
            missing_text = ", ".join(repr(ch) for ch in missing)
            raise ValueError(
                "Prompt contains characters outside the checkpoint vocab: "
                f"{missing_text}"
            )
        token_ids = loaded.vocab.encode(prompt_tail)
        return torch.tensor(token_ids, dtype=torch.long), prompt_tail, None

    source_path = data_path or loaded.train_config.data_path
    source_path = maybe_prepare_data(source_path, should_download=should_download)
    text = read_text(source_path)
    _, _, val_dataset = build_datasets(
        text=text,
        context_len=loaded.model_config.context_len,
        val_fraction=loaded.train_config.val_fraction,
    )
    inputs, targets = val_dataset[prompt_index]
    prompt_text = loaded.vocab.decode(inputs.tolist())
    actual_text = loaded.vocab.decode([int(targets[-1].item())])
    return inputs, prompt_text, actual_text


def render_trace_frames(
    trace: list[torch.Tensor],
    prompt_text: str,
    predicted_text: str,
    actual_text: str | None,
    seq_len: int,
    grid_width: int,
    cell_size: int,
) -> list[Image.Image]:
    frames = [state[0].permute(1, 2, 0).cpu() for state in trace]

    deltas = [torch.zeros_like(frames[0].norm(dim=-1))]
    for i in range(1, len(frames)):
        deltas.append((frames[i] - frames[i - 1]).norm(dim=-1))

    all_deltas = torch.stack(deltas[1:]) if len(deltas) > 1 else torch.stack(deltas)
    delta_max = float(all_deltas.max().item())
    delta_scale = max(delta_max, 1e-8)

    font = ImageFont.load_default()
    height, width = deltas[0].shape
    gap = 1
    margin_x = 20
    margin_y = 48
    margin_bottom = 12
    grid_pixel_w = width * cell_size + (width - 1) * gap
    grid_pixel_h = height * cell_size + (height - 1) * gap
    image_width = margin_x * 2 + grid_pixel_w
    image_height = margin_y + margin_bottom + grid_pixel_h
    last_token_index = max(seq_len - 1, 0)
    last_row = last_token_index // grid_width
    last_col = last_token_index % grid_width

    bg = (255, 255, 255)
    text_color = (55, 65, 81)
    dim_text = (156, 163, 175)
    border_color = (229, 231, 235)
    padding_fill = (243, 244, 246)
    readout_color = (245, 158, 11)

    rendered: list[Image.Image] = []
    for step_index, delta_grid in enumerate(deltas):
        image = Image.new("RGB", (image_width, image_height), color=bg)
        draw = ImageDraw.Draw(image)

        step_label = "step 0 (input)" if step_index == 0 else f"step {step_index}/{len(deltas) - 1}"
        draw.text((margin_x, 10), step_label, fill=text_color, font=font)

        detail = f"pred: {repr(predicted_text)}"
        if actual_text is not None:
            detail += f"  actual: {repr(actual_text)}"
        draw.text((margin_x, 26), detail, fill=dim_text, font=font)

        for row in range(height):
            for col in range(width):
                x0 = margin_x + col * (cell_size + gap)
                y0 = margin_y + row * (cell_size + gap)

                token_index = row * grid_width + col
                is_padding = token_index >= seq_len

                if is_padding:
                    fill = padding_fill
                elif step_index == 0:
                    fill = (245, 247, 250)
                else:
                    normalized = float(delta_grid[row, col].item() / delta_scale)
                    fill = color_map(normalized)

                draw.rectangle(
                    (x0, y0, x0 + cell_size, y0 + cell_size),
                    fill=fill,
                    outline=border_color,
                )

                if token_index < seq_len and cell_size >= 14:
                    label = display_char(prompt_text[token_index])
                    if not is_padding and step_index > 0:
                        normalized = float(delta_grid[row, col].item() / delta_scale)
                    else:
                        normalized = 0.0
                    label_color = (255, 255, 255) if normalized > 0.55 else text_color
                    draw.text((x0 + 3, y0 + 2), label, fill=label_color, font=font)

        fx0 = margin_x + last_col * (cell_size + gap)
        fy0 = margin_y + last_row * (cell_size + gap)
        draw.rectangle(
            (fx0 - 1, fy0 - 1, fx0 + cell_size + 1, fy0 + cell_size + 1),
            outline=readout_color,
            width=2,
        )
        rendered.append(image)

    return rendered


def main() -> None:
    args = parse_args()
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    model = loaded.model
    if not isinstance(model, CellularAutomataLanguageModel):
        raise ValueError("Visualization only supports CA checkpoints.")

    tokens, prompt_text, actual_text = resolve_example(
        loaded=loaded,
        prompt=args.prompt,
        prompt_index=args.prompt_index,
        data_path=args.data_path,
        should_download=args.download_dataset,
    )
    input_batch = tokens.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        logits, trace = model(input_batch, return_trace=True)

    predicted_text = loaded.vocab.decode([int(logits[:, -1, :].argmax(dim=-1).item())])
    frames = render_trace_frames(
        trace=trace,
        prompt_text=prompt_text,
        predicted_text=predicted_text,
        actual_text=actual_text,
        seq_len=tokens.size(0),
        grid_width=model.grid_width,
        cell_size=args.cell_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
    )

    summary_path = args.output.with_suffix(".txt")
    summary_path.write_text(
        "\n".join(
            [
                f"checkpoint: {args.checkpoint}",
                f"prompt: {repr(prompt_text)}",
                f"predicted_next: {repr(predicted_text)}",
                f"actual_next: {repr(actual_text) if actual_text is not None else 'n/a'}",
                f"grid: {model.grid_height}x{model.grid_width}",
                f"ca_steps: {model.num_steps}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved gif to: {args.output}")
    print(f"saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
