"""
Read-only inspection of a trained MAGNET checkpoint's learned segmentation
behavior: paper Table 4-style qualitative examples (predicted segment
boundaries marked with "||"), plus the actual cross-language equitability
metric — average segments/example and average bytes/segment across each
language's full eval_holdout set, to see whether languages are getting
comparable segmentation granularity or one is still over-fragmented.

Loads an existing checkpoint and runs forward passes only, in eval mode
(deterministic boundary thresholding at p > 0.5, not the stochastic
Gumbel-sigmoid sampling used during training) — does not train or modify
the model.

Usage:
    python -m src.eval.inspect_segmentation \\
        --checkpoint-path checkpoints/step_6000.pt --data-root /path/to/DATA_ROOT
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from src.model.boundary_predictor import gumbel_sigmoid_boundaries
from src.model.hourglass_transformer import _finalize_boundaries
from src.model.magnet import MAGNET
from src.training.collate import LANGUAGE_TO_ID, collate_batch
from src.training.dataset import LANGUAGES, VOCAB_SIZE, MagnetByteDataset, _corpus_path

DEFAULT_LANGUAGES = ("sw", "zu", "am")
NUM_EXAMPLES = 5


def resolve_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device):
    """
    Rebuilds the model architecture from the checkpoint's own saved config
    (dataclasses.asdict(TrainConfig) — see [[src.training.train.save_checkpoint]]),
    so this works for any checkpoint regardless of what model size it was
    trained with.
    """
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload["config"]
    model = MAGNET(
        vocab_size=VOCAB_SIZE,
        languages=LANGUAGES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers_tokenization=cfg["n_layers_tokenization"],
        n_layers_middle=cfg["n_layers_middle"],
        n_layers_final=cfg["n_layers_final"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("step")


def hard_boundaries_for(model, byte_ids, language_id, device):
    """
    Reconstructs the exact hard boundary decisions the model used
    internally — [[src.model.hourglass_transformer.HourglassTransformer.forward]]
    computes these but only returns the soft boundary_probs, so this
    re-derives them the same way: deterministic threshold (eval mode, no
    Gumbel noise) via [[gumbel_sigmoid_boundaries]], then the same
    finalization ([[_finalize_boundaries]]) the model applies before pooling.
    """
    input_ids = torch.tensor([byte_ids], dtype=torch.long, device=device)
    language_ids = torch.tensor([language_id], dtype=torch.long, device=device)
    with torch.no_grad():
        _, boundary_probs = model(input_ids, language_ids, attention_mask=None)
    hard = gumbel_sigmoid_boundaries(boundary_probs, training=False)
    hard = _finalize_boundaries(hard, attention_mask=None)
    return hard[0].tolist()


def render_segments(byte_ids, hard_boundaries):
    """
    Decodes the FULL byte sequence once — not each segment's raw bytes
    independently, which breaks for any script using multi-byte UTF-8
    characters (e.g. Geez, 3 bytes/char): a boundary falling mid-character
    would split a valid encoding across two invalid fragments. Instead,
    predicted byte-level boundaries are mapped onto the already-decoded
    text and "||" is inserted after whichever character each boundary
    byte falls within.

    n_segments is the true byte-level segment count (sum of
    hard_boundaries), matching [[full_eval_stats]]'s methodology exactly.
    This can exceed the number of "||"-separated chunks shown: when a
    boundary lands strictly inside a multi-byte character rather than
    exactly on a character edge, that's counted in the returned
    mid_char_splits — a genuinely interesting finding in its own right
    (the model splitting a single character across two segments), not
    swept under errors="replace" as before.
    """
    text = bytes(byte_ids).decode("utf-8", errors="replace")
    n_segments = int(sum(hard_boundaries))
    boundary_bytes = {i for i, is_boundary in enumerate(hard_boundaries) if is_boundary}

    char_byte_start = 0
    parts, current = [], []
    mid_char_splits = 0
    for ch in text:
        current.append(ch)
        char_byte_end = char_byte_start + len(ch.encode("utf-8"))  # exclusive
        boundary_in_char = any(b in boundary_bytes for b in range(char_byte_start, char_byte_end))
        if boundary_in_char:
            if (char_byte_end - 1) not in boundary_bytes:
                mid_char_splits += 1
            parts.append("".join(current))
            current = []
        char_byte_start = char_byte_end
    if current:
        parts.append("".join(current))

    return "||".join(parts), n_segments, mid_char_splits


def read_example_lines(eval_txt_path, n, max_bytes):
    """
    Pulls up to n non-empty lines from an eval_holdout file for qualitative
    display, truncated to max_bytes so each fits in a single forward pass.
    """
    lines = []
    with open(eval_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            byte_ids = list(line.encode("utf-8"))
            if len(byte_ids) < 2:
                continue
            lines.append(byte_ids[:max_bytes])
            if len(lines) >= n:
                break
    return lines


def print_examples(model, data_root, lang, language_id, device, max_seq_len, n):
    eval_path = _corpus_path(data_root, lang, "eval")
    lines = read_example_lines(eval_path, n, max_seq_len)
    print(f"\n=== {lang}: {len(lines)} example sentence(s) ===")
    for byte_ids in lines:
        hard = hard_boundaries_for(model, byte_ids, language_id, device)
        rendered, n_segments, mid_char_splits = render_segments(byte_ids, hard)
        mid_char_note = f", {mid_char_splits} mid-character" if mid_char_splits else ""
        print(f"  [{n_segments} segments{mid_char_note}, {len(byte_ids)} bytes] {rendered}")


@torch.no_grad()
def full_eval_stats(model, data_root, lang, language_id, device, max_seq_len, batch_size):
    """
    The actual equitability metric: average segments/example and average
    bytes/segment across a language's FULL eval_holdout set (not just the
    printed examples) — comparing these across languages shows whether
    segmentation granularity is comparable or one language is still being
    over-fragmented relative to the others.
    """
    dataset = MagnetByteDataset(data_root, [lang], split="eval", max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    total_segments, total_bytes, n_examples = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        language_ids = torch.full((input_ids.size(0),), language_id, dtype=torch.long, device=device)

        _, boundary_probs = model(input_ids, language_ids, attention_mask)
        hard = gumbel_sigmoid_boundaries(boundary_probs, training=False)
        hard = _finalize_boundaries(hard, attention_mask)

        total_segments += hard.sum().item()
        total_bytes += attention_mask.sum().item()
        n_examples += input_ids.size(0)

    avg_segments = total_segments / n_examples if n_examples else float("nan")
    avg_bytes_per_segment = total_bytes / total_segments if total_segments else float("nan")
    return {
        "n_examples": n_examples,
        "avg_segments_per_example": avg_segments,
        "avg_bytes_per_segment": avg_bytes_per_segment,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("MAGNET_DATA_ROOT", ""))
    parser.add_argument("--languages", type=str, default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    # This script's whole purpose is printing multi-script text (Latin,
    # Geez, ...) — force UTF-8 stdout so it doesn't crash on Windows
    # consoles that default to a codepage (e.g. cp1252) lacking those
    # characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if not args.data_root:
        raise SystemExit("Set --data-root or the MAGNET_DATA_ROOT environment variable.")

    device = resolve_device(args.device)
    languages = args.languages.split(",")

    model, step = load_model(args.checkpoint_path, device)
    print(f"loaded checkpoint at step {step} ({args.checkpoint_path}), device={device}")

    for lang in languages:
        print_examples(model, args.data_root, lang, LANGUAGE_TO_ID[lang], device, args.max_seq_len, args.num_examples)

    print("\n=== full eval_holdout segmentation stats (equitability check) ===")
    for lang in languages:
        stats = full_eval_stats(
            model, args.data_root, lang, LANGUAGE_TO_ID[lang], device, args.max_seq_len, args.eval_batch_size
        )
        print(
            f"  {lang}: n={stats['n_examples']} "
            f"avg_segments/example={stats['avg_segments_per_example']:.2f} "
            f"avg_bytes/segment={stats['avg_bytes_per_segment']:.2f}"
        )


if __name__ == "__main__":
    main()
