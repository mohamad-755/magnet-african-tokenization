# MAGNET: African Tokenization Fairness

## Project description

This project reproduces and extends MAGNET (Ahia et al., NeurIPS 2024), a
gradient-based tokenization method that uses language-script-specific
boundary predictors to achieve equitable segmentation across languages.
MAGNET's original evaluation covered 9 languages across Latin, Cyrillic, and
Indic scripts — no African languages were tested. This project evaluates
MAGNET on 9 African languages: Amharic (am), Hausa (ha), Igbo (ig),
Chichewa/Nyanja (ny), Kinyarwanda (rw), Swahili (sw), Wolof (wo), Yoruba
(yo), Zulu (zu) — spanning four language families (Niger-Congo/Bantu,
Afro-Asiatic, Niger-Congo/Volta-Niger, Niger-Congo/Atlantic) to test whether
results generalize rather than overfitting to one morphological pattern.
Since most of these languages share Latin script (with Amharic as the
exception, using Ge'ez script), this tests whether MAGNET's script-level
boundary predictor design — built for structurally distinct scripts —
actually achieves fairness across morphologically diverse languages that
happen to share a script, or whether finer-grained adaptation is needed.
See data_card.md for full corpus documentation, cleaning methodology, and
known limitations.

## Background

- Ahia et al. 2024, MAGNET (NeurIPS), https://openreview.net/forum?id=1e3MOwHSIX
- Petrov et al. 2023, "Language Model Tokenizers Introduce Unfairness Between
  Languages" (NeurIPS), https://arxiv.org/abs/2305.15425

## Project status

Corpus cleaning complete for all 9 languages (see data_card.md). MAGNET
architecture reproduction in progress.

## Repo structure

- `data/raw/` — raw, unprocessed corpus data per language (gitignored)
- `data/cleaned/` — cleaned/preprocessed corpus data ready for tokenizer
  training (gitignored)
- `src/model/` — MAGNET model architecture code
- `src/training/` — training loops and tokenizer fitting scripts
- `src/eval/` — tokenization fairness evaluation and metrics
- `configs/` — language metadata and experiment configuration files
- `notebooks/` — exploratory analysis and visualization notebooks
- `results/` — experiment outputs, metrics, and figures (gitignored)
- `data_card.md` — corpus documentation, cleaning methodology, and known
  limitations

## Acknowledgments

Developed as part of LebNet Tech Fellows, Option 1: Academic Paper
Reproduction + Extension.
