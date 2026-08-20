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

Corpus cleaning complete for all 9 languages (see data_card.md). MAGNET is
reproduced (`src/model/`) and trained (`src/training/train.py`), with three
completed comparisons and one documented negative result:

- **Script-level vs. per-language boundary-predictor routing** (both at a
  uniform target compression rate, β=0.5): giving each language its own
  predictor instead of sharing one per script did not improve
  cross-language fairness — coefficient of variation in bytes/segment was
  about 1.4x higher under per-language routing (0.106 vs. 0.074), and
  isolated to just the 8 Latin-script languages (removing Amharic's
  confound, since it already had a dedicated predictor either way) about
  1.8x higher (0.112 vs. 0.061). Sharing weights across languages appears
  to act as an implicit consistency mechanism that independent
  per-language predictors don't get for free.
  See `results/per_language_vs_script_level_comparison.json`.
- **MAGNET vs. a BPE baseline**: comparing bytes/segment and its
  coefficient of variation across languages, see
  `results/bpe_baseline_stats.json`. MAGNET's raw byte-level boundaries can
  split individual multi-byte characters (severely for Amharic/Ge'ez, ~30%+
  of segments in some samples); BPE cannot, by construction.
- **Per-language β targets (paper Eq. 4) — training collapse, unresolved.**
  Computing each language's own target compression rate from its
  byte-to-word ratio and training with those 9 distinct values (rather than
  one shared β=0.5) caused the boundary predictor to collapse to
  effectively zero real boundaries, uniformly across all 9 languages
  regardless of how different their individual targets were — evidence of
  a shared optimization failure (likely the regularizer dominating the LM
  loss early and saturating the boundary predictor toward the trivial
  solution), not of genuine per-language convergence. Reducing
  `reg_weight` 10x (1.0 → 0.1) did not resolve it. Not solved within this
  project's time budget — the interaction between per-language β spread
  and regularizer weighting needs further study.

## Limitations / known open issues

- The per-language β training collapse above is unresolved; `--beta-by-language`
  values far from 0.5 are not currently safe to train with at
  `reg_weight >= 0.1` without further investigation.
- The "we tried β=(0.157, 0.081) under script-level routing and it doesn't
  work" claim (referenced during development) was never independently
  reproduced with visible output in this project's own history — worth
  re-verifying if it becomes relevant again.

## Repo structure

- `data/` — see `data/README.md` for the external DATA_ROOT link
  (corpora + all experiment checkpoints) and structure. Training/eval read
  directly from it via `--data-root`/`MAGNET_DATA_ROOT`; `data/raw/` is a
  gitignored fallback spot for local copies of any corpus file that turns
  out unreliable to read straight from Drive (not a general mirror)
- `src/model/` — MAGNET model architecture code (`magnet.py`,
  `hourglass_transformer.py`, `boundary_predictor.py`, `losses.py`)
- `src/training/` — data pipeline and training loop (`dataset.py`,
  `collate.py`, `train.py`)
- `src/eval/` — evaluation and analysis scripts: `inspect_segmentation.py`
  (checkpoint segmentation inspection), `compute_beta.py` (paper Eq. 4
  target compression rate), `bpe_baseline.py` (BPE comparison),
  `compare_routing_designs.py` (script-level vs. per-language comparison)
- `configs/` — language metadata and experiment configuration files
- `notebooks/` — exploratory analysis and visualization notebooks
- `results/` — experiment outputs (gitignored except the two comparison
  JSONs explicitly tracked: `bpe_baseline_stats.json`,
  `per_language_vs_script_level_comparison.json`)
- `data_card.md` — corpus documentation, cleaning methodology, and known
  limitations

## Acknowledgments

Developed as part of LebNet Tech Fellows, Option 1: Academic Paper
Reproduction + Extension.
