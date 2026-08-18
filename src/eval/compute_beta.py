"""
Computes MAGNET's target compression rate beta (Ahia et al., 2024, Eq. 4,
Section 2.2) for one or more anchor languages: beta = 1 / R_bar.

R_bar is the average, over documents in the language's train/corpus.txt
(one per line), of each document's OWN byte-to-word ratio:

    R_bar = (1/D) * sum(|x_i| / countwords(x_i))  for i = 1..D

— not a single corpus-wide ratio of total bytes to total words. The two
diverge whenever document length varies a lot (a corpus-level aggregate
is dominated by its longest documents; the per-document average weights
every document, including short stubs, equally).

Read-only: does not train or modify anything, just reports numbers so a
value can be reviewed before being passed to train.py's --beta-by-script.

Usage:
    python -m src.eval.compute_beta --languages sw,am --data-root /path/to/DATA_ROOT
"""
import argparse
import os
import sys

from src.training.dataset import _corpus_path


def compute_r_bar_aggregate(corpus_path):
    """
    Corpus-level ratio of sums: total UTF-8 bytes / total whitespace-split
    words across the whole file. NOT the paper's Eq. 4 — kept only so it
    can be reported side by side with [[compute_r_bar_per_sequence]] to
    show how much the two diverge.
    """
    total_bytes = 0
    total_words = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            total_bytes += len(line.encode("utf-8"))
            total_words += len(line.split())
    if total_words == 0:
        raise ValueError(f"{corpus_path} has zero whitespace-split words — can't compute R_bar")
    return total_bytes / total_words, total_bytes, total_words


def compute_r_bar_per_sequence(corpus_path):
    """
    Paper's actual Eq. 4: the average of each document's OWN byte-to-word
    ratio, one document per line. A document's byte length excludes its
    trailing line-separator (stripped before encoding) since that's a file
    artifact, not corpus content. Documents with zero whitespace-split
    words (a ratio would be undefined) are skipped defensively, though the
    >=15-word minimum filter already applied during cleaning
    (data_card.md Section 4, Rule 1) means this shouldn't occur in practice.
    """
    ratios = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_words = len(line.split())
            if n_words == 0:
                continue
            ratios.append(len(line.encode("utf-8")) / n_words)
    if not ratios:
        raise ValueError(f"{corpus_path} produced no valid per-document ratios")
    return sum(ratios) / len(ratios), len(ratios)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", type=str, required=True, help="comma-separated language codes, e.g. sw,am")
    parser.add_argument("--data-root", type=str, default=os.environ.get("MAGNET_DATA_ROOT", ""))
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if not args.data_root:
        raise SystemExit("Set --data-root or the MAGNET_DATA_ROOT environment variable.")

    for lang in args.languages.split(","):
        corpus_path = _corpus_path(args.data_root, lang, args.split)

        agg_r_bar, total_bytes, total_words = compute_r_bar_aggregate(corpus_path)
        agg_beta = 1.0 / agg_r_bar

        seq_r_bar, n_docs = compute_r_bar_per_sequence(corpus_path)
        seq_beta = 1.0 / seq_r_bar

        print(f"{lang}:")
        print(
            f"  [aggregate, NOT paper Eq. 4] bytes={total_bytes} words={total_words} "
            f"R_bar={agg_r_bar:.4f} -> beta={agg_beta:.4f}"
        )
        print(
            f"  [per-sequence, paper Eq. 4]  documents={n_docs} "
            f"R_bar={seq_r_bar:.4f} -> beta={seq_beta:.4f}"
        )


if __name__ == "__main__":
    main()
