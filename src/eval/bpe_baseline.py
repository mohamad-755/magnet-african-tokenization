"""
BPE tokenizer baseline for comparison against MAGNET's learned segmentation.

Trains one per-language SentencePiece BPE model on train/corpus.txt,
evaluates average bytes/segment on eval_holdout (aggregate ratio: total
eval bytes / total tokens, matching how MAGNET's own bytes/segment is
computed in inspect_segmentation.py's full_eval_stats), and compares
fairness — coefficient of variation of bytes/segment across the 9
languages — against MAGNET's results.

MAGNET_STATS below is NOT re-derived by this script. It's taken directly
from the eval_holdout inspection output already produced by running
inspect_segmentation.py against checkpoints_all_9langs/step_4999.pt (a
100%-Colab-side run, uniform beta_by_script=(0.5, 0.5) across scripts) —
reusing those already-verified numbers rather than re-loading that
checkpoint locally, given this project's repeated, well-documented
Google-Drive read-reliability issues specifically around the checkpoints/
directory.

Usage:
    python -m src.eval.bpe_baseline --data-root /path/to/DATA_ROOT
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import sentencepiece as spm

from src.training.dataset import LANG_TO_FOLDER, _corpus_path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "bpe_baseline_stats.json"
BPE_MODEL_DIR = REPO_ROOT / "results" / "bpe_models"
LOCAL_HA_OVERRIDE = REPO_ROOT / "data" / "raw" / "ha_wikipedia" / "train" / "corpus.txt"

# From the user's real Colab run: checkpoints_all_9langs/step_4999.pt,
# uniform beta_by_script=(0.5, 0.5), full eval_holdout inspection.
MAGNET_CHECKPOINT_INFO = {
    "checkpoint_dir": "checkpoints_all_9langs",
    "step": 4999,
    "beta_by_script": [0.5, 0.5],
}
MAGNET_STATS = {
    "sw": {"n_examples": 6344, "avg_segments_per_example": 218.85, "avg_bytes_per_segment": 2.34},
    "zu": {"n_examples": 746, "avg_segments_per_example": 227.55, "avg_bytes_per_segment": 2.25},
    "yo": {"n_examples": 1930, "avg_segments_per_example": 251.61, "avg_bytes_per_segment": 2.03},
    "ig": {"n_examples": 12258, "avg_segments_per_example": 222.74, "avg_bytes_per_segment": 2.30},
    "ha": {"n_examples": 25518, "avg_segments_per_example": 201.15, "avg_bytes_per_segment": 2.55},
    "ny": {"n_examples": 113, "avg_segments_per_example": 217.13, "avg_bytes_per_segment": 2.35},
    "am": {"n_examples": 1933, "avg_segments_per_example": 261.52, "avg_bytes_per_segment": 1.96},
    "rw": {"n_examples": 1407, "avg_segments_per_example": 235.37, "avg_bytes_per_segment": 2.17},
    "wo": {"n_examples": 307, "avg_segments_per_example": 223.45, "avg_bytes_per_segment": 2.29},
}
LANGUAGES = tuple(LANG_TO_FOLDER.keys())  # sw, ha, ig, ny, rw, sw, wo, yo, zu order from dataset.py


def train_bpe(corpus_path, model_prefix, vocab_size, input_sentence_size=1_000_000):
    """
    Trains a SentencePiece BPE model. character_coverage=1.0 regardless of
    corpus size (rather than the <1.0 typically used for very large
    multilingual corpora) so rare Geez characters in Amharic aren't
    silently dropped/mapped to <unk> — full coverage is entirely tractable
    at this project's monolingual, tens-of-MB-at-most corpus sizes.
    input_sentence_size caps training time on the larger corpora (Hausa,
    Igbo); shuffled first so the cap doesn't just sample the file's start.
    """
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
    )


def evaluate_bpe(model_path, eval_txt_path):
    """
    Aggregate ratio (total bytes / total tokens across the whole eval
    file), matching [[inspect_segmentation.full_eval_stats]]'s methodology
    for MAGNET, so the two are directly comparable.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    total_bytes, total_tokens = 0, 0
    with open(eval_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_bytes += len(line.encode("utf-8"))
            total_tokens += len(sp.encode(line, out_type=int))
    avg_bytes_per_segment = total_bytes / total_tokens if total_tokens else float("nan")
    return avg_bytes_per_segment, total_bytes, total_tokens


def coefficient_of_variation(values):
    """Population stdev / mean — lower = more consistent (fairer) across languages."""
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean


def run_language(lang, data_root, vocab_size):
    train_path = _corpus_path(data_root, lang, "train")
    if lang == "ha" and LOCAL_HA_OVERRIDE.exists():
        train_path = LOCAL_HA_OVERRIDE  # avoid Drive's known large-file read failures for this file
    eval_path = _corpus_path(data_root, lang, "eval")

    BPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_prefix = BPE_MODEL_DIR / f"{lang}_bpe"
    train_bpe(train_path, model_prefix, vocab_size)

    avg_bytes_per_segment, total_bytes, total_tokens = evaluate_bpe(f"{model_prefix}.model", eval_path)
    return {
        "avg_bytes_per_segment": avg_bytes_per_segment,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=os.environ.get("MAGNET_DATA_ROOT", ""))
    parser.add_argument("--languages", type=str, default=",".join(LANGUAGES))
    parser.add_argument("--vocab-size", type=int, default=8000)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if not args.data_root:
        raise SystemExit("Set --data-root or the MAGNET_DATA_ROOT environment variable.")

    languages = args.languages.split(",")

    bpe_stats = {}
    for lang in languages:
        print(f"training BPE for {lang} (vocab_size={args.vocab_size})...")
        bpe_stats[lang] = run_language(lang, args.data_root, args.vocab_size)
        print(f"  {lang}: {bpe_stats[lang]}")

    magnet_values = [MAGNET_STATS[l]["avg_bytes_per_segment"] for l in languages]
    bpe_values = [bpe_stats[l]["avg_bytes_per_segment"] for l in languages]
    magnet_cv = coefficient_of_variation(magnet_values)
    bpe_cv = coefficient_of_variation(bpe_values)

    results = {
        "magnet_checkpoint": MAGNET_CHECKPOINT_INFO,
        "bpe_vocab_size": args.vocab_size,
        "languages": languages,
        "magnet": {l: MAGNET_STATS[l] for l in languages},
        "magnet_coefficient_of_variation": magnet_cv,
        "bpe": bpe_stats,
        "bpe_coefficient_of_variation": bpe_cv,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nresults saved: {RESULTS_PATH}")

    print(f"\n{'lang':<6}{'MAGNET bytes/seg':<20}{'BPE bytes/seg':<18}")
    for l in languages:
        print(f"{l:<6}{MAGNET_STATS[l]['avg_bytes_per_segment']:<20.4f}{bpe_stats[l]['avg_bytes_per_segment']:<18.4f}")
    print(f"\ncoefficient of variation (fairness, lower = more equitable):")
    print(f"  MAGNET (uniform beta=0.5): {magnet_cv:.4f}")
    print(f"  BPE (vocab={args.vocab_size}):          {bpe_cv:.4f}")


if __name__ == "__main__":
    main()
