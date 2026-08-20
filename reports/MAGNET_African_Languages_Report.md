# MAGNET on African Languages: Script-Level vs. Per-Language Boundary Predictors

## 1. Project Title & Abstract

MAGNET (Ahia et al., NeurIPS 2024) learns tokenization boundaries via gradient-based, per-script boundary predictors, but was only evaluated on Latin, Cyrillic, and Indic scripts — no African languages. This project reproduces MAGNET on 9 African languages (Latin and Geez scripts, 4 language families) and extends it with a per-language boundary predictor variant, testing whether routing by language — not just script — improves cross-language segmentation fairness.

It doesn't. Per-language routing raises the coefficient of variation (CV) in bytes/segment from 0.074 to 0.106 — less fair, not more. A second extension, using the paper's own Eq. 4 formula to give each language its own target compression rate, collapsed training outright: every language converged to zero real segment boundaries, regardless of regularizer weight. Both are real, reproducible, negative results, reported honestly alongside a working BPE baseline comparison.

## 2. Introduction & Problem Statement

Standard tokenizers systematically disadvantage morphologically complex and non-Latin-script languages: the same content costs more tokens, more compute, and often worse downstream model quality (Petrov et al., 2023). MAGNET addresses this by learning segmentation boundaries during training rather than fixing a vocabulary upfront, using a boundary predictor per script. But 8 of our 9 target languages share Latin script despite very different morphology (Bantu agglutination, Chadic and Volta-Niger patterns, Atlantic patterns) — so script-level routing may be too coarse a signal. This project asks: does MAGNET's script-level design generalize to African languages, and does routing by language instead of script actually help?

## 3. Methodology

**Data:** 9 languages — Amharic, Hausa, Igbo, Chichewa, Kinyarwanda, Swahili, Wolof, Yoruba, Zulu — Wikipedia-sourced (Wolof supplemented with MasakhaNER for its training split), cleaned and held out per `data_card.md`. All use Latin script except Amharic (Geez).

**Architecture:** MAGNET's hourglass transformer (2/6/2 layers, d_model=256, 4 heads), Gumbel-sigmoid boundary sampling (paper Eq. 1–2), and a binomial regularizer (Eq. 3) toward a target compression rate β. We implemented both the paper's script-routed boundary predictor (one per script) and our own extension, a language-routed variant (one predictor per language).

**β calibration:** The paper's Eq. 4 defines β = 1/R̄, where R̄ is the average, across documents, of each document's own byte-to-word ratio. We initially computed this as a single corpus-wide ratio instead of a per-document average — a real bug, caught and fixed before it reached any training run. The corrected computation was validated against `data_card.md`'s own document counts (exact match for Swahili: 82,523; Amharic: 8,096).

**Training:** Adam (β=(0.9, 0.98), eps=1e-6), lr 5e-5, 5000 steps, batch size 16 — a reduced budget given project time constraints, not the paper's full schedule. All reported runs are seeded for reproducibility.

## 4. Implementation Details & Results

**Script-level routing, uniform β=0.5** (the paper's design): trains stably. Bytes/segment ranges 1.96–2.55 across all 9 languages, CV = 0.074.

**Per-language routing, uniform β=0.5** (our extension): also trains stably, but the range widens to 1.69–2.31, CV = 0.106 — worse fairness than script-level routing. Isolating the 8 Latin-script languages (which went from sharing one predictor under script-level routing to each having their own) makes the gap larger, not smaller: CV 0.061 → 0.112.

**Per-language routing, per-language β** (paper's Eq. 4 target, computed per language): collapses. At reg_weight=1.0, and again at reg_weight=0.1 (10x weaker), every language converges to exactly 1 segment per 512-byte example. The regularizer's asymmetric penalty — −log(β) for placing a boundary vs. −log(1−β) for not placing one — makes "don't segment" far cheaper once β is small, and the model takes that shortcut regardless of regularizer weight. Both runs land on bit-identical eval statistics, which is itself informative: once segmentation fully collapses, bytes/segment stops depending on the model at all and just reflects the data's own average example length.

**BPE baseline:** a per-language SentencePiece BPE tokenizer (vocab=8000) gets CV = 0.138 on the same metric — nearly double MAGNET's script-level CV. MAGNET's gradient-based approach is more cross-language-consistent than fixed-vocabulary BPE here, though not for free: MAGNET's raw byte boundaries can and do split individual multi-byte Geez characters mid-character, something BPE cannot do by construction.

Full numbers: `results/per_language_vs_script_level_comparison.json`, `results/bpe_baseline_stats.json`.

## 5. Discussion & Analysis

The main finding is negative, and that's the finding worth reporting: giving each language its own boundary predictor doesn't improve fairness, and using the paper's own per-language target-rate formula actively breaks training. Our read is that sharing one predictor across languages was quietly doing some of the fairness work itself — plausibly by pooling gradient signal across languages through a shared module — and splitting predictors apart removed that benefit, even though intuitively it should have let each language specialize further.

The β collapse remains unresolved. We ruled out regularizer weight as the sole cause (cutting it 10x didn't help), which narrows the problem but doesn't solve it. A regularizer warm-up — ramping β's influence in gradually instead of applying it at full strength from step 0 — is our best untested hypothesis.

**Limitations:** results come from a single seed and a reduced training budget (5000 steps vs. a full schedule); Chichewa (113 eval examples) and Wolof (307) have noticeably smaller eval sets than the rest; no downstream task evaluation (translation, NER) was run — everything here is a segmentation-statistic proxy for fairness, not a measurement of end-to-end model quality.

## 6. Reflection on Learnings

The hardest part of this project wasn't the model architecture — it was learning to distrust my own numbers. I caught myself computing the paper's target compression rate wrong, averaging bytes and words across the whole corpus instead of per document, the way Eq. 4 actually specifies. It took a second pass to notice the two methods gave meaningfully different results, not just rounding noise. Later, I had a full results write-up drafted early, and when I checked it line by line against what I'd actually run, a large chunk of it turned out to be invented — experiments that never happened, a language's script labeled wrong, numbers that contradicted each other in the same document. Rebuilding it from only verified numbers taught me more about doing careful research than any of the model debugging did.

The most rewarding moment was the opposite kind of surprise. When I gave each language its own target compression rate instead of sharing one across all of them, the model didn't get better at treating languages differently — it just stopped segmenting entirely, for every language, regardless of how different their targets were supposed to be. Cutting the regularizer's weight by 10x didn't fix it. I don't have a clean answer for why yet, and I had to be okay submitting that as an open question instead of a solved one.

The most tedious part had nothing to do with machine learning at all: one language's raw text file kept failing to read from Google Drive, for reasons that took real effort to even diagnose, let alone fix. Good reminder that a lot of real research time doesn't go into the interesting part.

If I did this again, I'd budget more time for hyperparameter exploration on the collapse before accepting it as unsolved, and I'd fact-check any generated writeup against my own logs from the first draft, not after the fact.

---

**Repository:** `magnet-african-tokenization` (public)
**Data + checkpoints:** https://drive.google.com/drive/folders/1jN8eXuBZ5IxXoGHM-zPz9im6IajAd8ca
**Reproducing these results:** `notebooks/00_colab_training_full_pipeline.ipynb`
