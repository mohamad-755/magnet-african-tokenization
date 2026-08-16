# Data Card â African Language Wikipedia Corpus for MAGNET Tokenization Fairness

**Project:** MAGNET: African Tokenization Fairness
**Status:** All 9 target languages complete (Swahili, Chichewa, Zulu,
Kinyarwanda, Hausa, Amharic, Yoruba, Wolof, Igbo). Wolof required a
second, supplementary source beyond Wikipedia to reach an adequate
corpus size; this is documented in Sections 2.1 and 6.

---

## 1. Motivation

Standard tokenizers â including gradient-based methods like MAGNET
(Ahia et al., 2024) â are typically developed and evaluated on a narrow
set of scripts and languages. MAGNET's original evaluation covered nine
languages across Latin, Cyrillic, and Indic scripts; no African languages
were tested. This corpus was collected to evaluate whether MAGNET's
script-level boundary predictor design generalizes to African languages
spanning different morphological structures (Bantu agglutination,
Volta-Niger and West Atlantic tonal marking, Chadic concatenative
morphology, Semitic root-and-pattern morphology with a non-Latin script) â
in particular, whether a single shared Latin-script predictor achieves
equitable segmentation across languages that share a script but differ
substantially in morphology.

## 2. Composition

| Language | Family | Script | Source | Raw extracted | Final cleaned | Retention |
|---|---|---|---|---|---|---|
| Swahili (sw) | Bantu | Latin | Wikipedia | 150,426 | 86,866 | 57.7% |
| Zulu (zu) | Bantu | Latin | Wikipedia | 14,273 | 3,859 | 27.0% |
| Yoruba (yo) | Volta-Niger | Latin | Wikipedia | 47,996 | 13,833 | 28.8% |
| Igbo (ig) | Volta-Niger | Latin | Wikipedia | 57,457 | 53,686 | 93.4% |
| Hausa (ha) | Chadic | Latin | Wikipedia | 114,318 | 104,631 | 91.5% |
| Chichewa (ny) | Bantu | Latin | Wikipedia | 1,408 | 892 | 63.4% |
| Amharic (am) | Semitic | Ethiopic | Wikipedia | 23,189 | 8,522 | 36.8% |
| Kinyarwanda (rw) | Bantu | Latin | Wikipedia | 13,850 | 11,125 | 80.3% |
| Wolof (wo) â  | West Atlantic | Latin | Wikipedia + MasakhaNER | 2,351 | 3,721 | see 2.1 |
| **Total** | | | | **425,268** | **287,135** | mixed sources â see per-row |

All Wikipedia-sourced languages draw from `dumps.wikimedia.org` monthly
snapshots, licensed `cc-by-sa-4.0` (text only). Retention rate varies
substantially by language â this reflects genuine differences in how each
language's Wikipedia is composed (see Section 6), not inconsistent
cleaning.

â  Wolof's "Raw extracted" and "Retention" columns reflect the Wikipedia
portion only. "Final cleaned" (3,721) is the combined total after adding a
supplementary source â see breakdown below.

### 2.1 Wolof multi-source breakdown

Wolof Wikipedia alone (1,073 documents, 405,389 words after cleaning) fell
well short of every other language in this corpus and was flagged during
spot-check as likely insufficient before cleaning even confirmed it. It
was supplemented with MasakhaNER 1.0's Wolof split (Adelani et al., 2021;
`github.com/masakhane-io/masakhane-ner`), a human-annotated NER dataset
sourced from local news articles, licensed `CC-BY-4.0-NC`. NER tags were
discarded and sentences reconstructed as plain text; train/dev/test splits
were combined since the original task split is irrelevant once the data
is used as raw tokenizer training/eval text rather than labeled NER
examples.

| Source | Domain | Documents/Sentences | Final words |
|---|---|---|---|
| Wikipedia | Encyclopedic | 1,073 | 405,389 |
| MasakhaNER 1.0 | News (human-annotated) | 2,648 | 52,777 |
| **Combined** | | **3,721** | **458,166** |

Even combined, Wolof's total word count (458,166) remains well below
Kinyarwanda's (1,931,619) and every other language except Chichewa â
documented as a genuine, unresolved shortfall rather than papered over by
the second source.

## 3. Collection Process

Each language followed an identical procedure, so cross-language
comparisons reflect real linguistic/data differences rather than uneven
handling. Wolof is the one documented exception (Section 2.1):

1. Register the source (license, access method, expected coverage) before
   any data was pulled.
2. Download the full Wikipedia dump; extract clean article text via
   `wikiextractor`, explicitly excluding non-article namespaces (categories,
   talk pages, user pages).
3. Manually spot-check a random sample of 20 extracted articles per
   language (two independent samples were drawn for Swahili, Zulu, and
   Yoruba) against three criteria: language correctness, encoding
   integrity, and diacritic/spelling intactness.
4. Apply an 8-rule automated cleaning pipeline (Section 4) informed
   directly by what the manual spot-checks found, and re-applied
   retroactively to earlier languages whenever a new rule was added.
5. **Wolof only:** after cleaning confirmed the Wikipedia corpus was
   substantially smaller than every other language's, source and convert
   a supplementary corpus (MasakhaNER 1.0) following a parallel but
   distinct procedure suited to its CoNLL-tagged format (Section 2.1),
   then combine both sources into one corpus before the holdout split.
6. Hold out a 5% evaluation split per language, stored in a separate
   directory from training data (`eval_holdout/`), with a recorded
   document count and SHA-256 content hash to verify it is never touched
   during training. For Wolof, the eval split was drawn from the combined
   (Wikipedia + MasakhaNER) corpus, not the Wikipedia-only corpus, so
   train and eval reflect the same source/domain mix.

## 4. Cleaning Rules Applied

Rules 1â8 below were applied identically across all Wikipedia-sourced
languages, including Kinyarwanda and the Wikipedia portion of Wolof:

1. **Minimum word count filter** (<15 words dropped) â addresses empty
   infobox/list-only page extractions, found in every language's
   spot-check sample. Severity varies substantially by language (10% of
   Igbo/Hausa's sample vs. 50% of Amharic's vs. 60% of Wolof's â see
   Section 6).
2. **Encoding sanity check** â drops documents matching known mojibake
   patterns. Zero triggered for Kinyarwanda; zero for Wolof.
3. **Exact deduplication** â content-hash based, catches verbatim repeats.
4. **Near-duplicate template capping** â masks numbers/proper nouns into a
   "template skeleton"; keeps at most 3 examples of any repeated skeleton.
   Motivated by Yoruba's spot-check (10 of 20 sampled articles were
   near-identical bot-generated asteroid-catalog stubs undetected by
   exact-hash dedup). Verified removal: 198 documents in Yoruba, 323 in
   Amharic, 86 in Kinyarwanda (21 distinct templates), 7 in Wolof (4
   distinct templates â including a newly observed pattern of
   near-identical stub articles about unrelated Lithuanian football clubs,
   a different template family than Yoruba's astronomy stubs but the same
   underlying risk).
5. **Outlier-length truncation** (99th percentile cap) â guards against a
   small number of extreme-length documents distorting downstream
   tokenizer/model training.
6. **Leaked XML revision-metadata removal** â found in Zulu's spot-check:
   raw, HTML-entity-encoded Wikipedia revision metadata leaking into
   extracted text via a `wikiextractor` edge case. Frequency varied hugely
   by language: 1-2 instances in Zulu/Chichewa, 11 in Kinyarwanda, 0 in
   Wolof, but **505 instances in Amharic** â a genuinely large gap worth
   flagging, though its underlying cause is not established (see
   Section 6). Fix verified correct at every observed frequency.
7. **Leaked templatestyles/infobox markup removal** â found in Igbo's
   spot-check: two structural variants (a clean tag-pair form, and a
   messier raw JSON "data-mw" form with no reliable closing boundary).
   The messier variant is a documented best-effort fix: it reliably
   removes the JSON debris but may occasionally remove a few adjacent
   real words with no whitespace separator (affects an estimated
   ~5/57,457 Igbo documents, ~0.01%). 50 instances found in Kinyarwanda;
   1 in Wolof.
8. **Citation marker removal** â strips leftover `[1]`, `[2][3]` footnote
   references not fully removed by extraction. 168 instances in
   Kinyarwanda; 11 in Wolof.

**Wolof's MasakhaNER portion used a separate, narrower procedure** (not
rules 1â8): a minimum sentence length of 3 words (vs. 15 for Wikipedia
articles, since these are single news sentences rather than full
articles) and exact deduplication only. Rules 2, 6, and 7 (encoding sanity,
XML/templatestyles leaks) are Wikipedia-extraction-specific artifacts and
do not apply to this source, so they were not run against it.

## 5. Evaluation Split

| Language | Train docs | Eval docs | Eval % |
|---|---|---|---|
| Swahili | 82,523 | 4,343 | 5.0% |
| Zulu | 3,667 | 192 | 5.0% |
| Yoruba | 13,142 | 691 | 5.0% |
| Igbo | 51,002 | 2,684 | 5.0% |
| Hausa | 99,400 | 5,231 | 5.0% |
| Chichewa | 848 | 44 | 4.9% |
| Amharic | 8,096 | 426 | 5.0% |
| Kinyarwanda | 10,569 | 556 | 5.0% |
| Wolof (combined) | 3,535 | 186 | 5.0% |

Each split uses a fixed seed (42) for reproducibility, and each eval set
has a recorded SHA-256 content hash (in its `eval_manifest.md`) so anyone
can verify a given corpus file was never touched during training.

## 6. Known Limitations

- **Empty-page extraction rates vary hugely by language and are a real
  property of each source Wikipedia, not a pipeline defect.** Ranges from
  ~5% (Kinyarwanda) and ~7% (Igbo, Hausa) to **50% (Amharic)** and **60%
  (Wolof)** in manually-verified 20-article samples. Amharic's failures
  were disproportionately bare year-number articles (e.g. "229 á¥.á¤.á .",
  "1938") â infobox-only date pages with zero prose. Wolof's failures were
  a mix of true empty stubs (35% of its sample) plus several near-threshold
  short stubs, without Amharic's specific date-page pattern.
- **Wolof required a supplementary source and remains the smallest corpus
  by word count even after combining sources** (458,166 words, vs.
  Chichewa's 204,640 and every other language's 1M+). Documented as a
  genuine shortfall rather than papered over.
- **Chichewa Wikipedia is extremely small**: only 1,408 raw articles,
  204,640 words after cleaning. Documented honestly as a genuine
  limitation for this language's downstream evaluation.
- **Tone/diacritic marking is inconsistently applied within the Yoruba and
  Igbo corpora.** Confirmed directly across independent samples: the same
  word appears both with and without diacritics in different articles
  (e.g. Igbo "amá»¥rá»¥" vs. "amuru" for "born"). Cannot be corrected
  automatically without a real morphological analyzer or native-speaker
  review. Hausa, Amharic, Kinyarwanda, and Wolof showed no equivalent
  pattern in their own orthographic systems in spot-check samples.
- **Leaked XML revision-metadata frequency is unexplained and highly
  uneven**: 505 instances in Amharic vs. 0â11 in every other language. The
  fix is verified correct regardless of frequency, but the cause of this
  concentration in Amharic specifically has not been investigated further.
- **A small number of Zulu articles (1 confirmed instance) contained
  informal/off-register content** (a "Polandball" internet-meme article
  ending in a stray "omg" fragment). Not a technical defect; noted for
  register/source_type tracking.
- **One unresolved, low-confidence signal remains in the Igbo corpus**: a
  single possible leftover citation-marker instance (1/53,686 documents,
  ~0.002%) that could not be located via direct text search across
  multiple verification attempts. Documented rather than pursued further.
- **One Kinyarwanda article in the spot-check sample showed an apparent
  mid-article code-switch into untranslated English** (a sentence that
  trails off into English text partway through). Likely a genuine
  Wikipedia editorial artifact (an unfinished translation) rather than an
  extraction bug; not currently addressed by an automated rule since it
  was observed once in a 20-document sample and not confirmed as a
  systemic pattern at full-corpus scale.

## 7. Cross-Language Observations

- **Non-Latin script (Amharic/Ethiopic) introduced no encoding defects.**
  Zero mojibake, zero wrong-language leaks across the full spot-check
  sample â the pipeline's encoding handling generalized correctly to a
  completely different writing system with no code changes required. The
  same holds for Kinyarwanda and Wolof's Latin-script diacritics (`Ã«`,
  `Ã `, `Ã±`, `Å`, apostrophe-marked forms), which rendered cleanly with
  zero encoding issues in both languages' spot-checks.
- **Retention rate is not correlated with corpus size.** Hausa (114,318
  raw) and Igbo (57,457 raw) both retained >90%, while Zulu (14,273 raw)
  and Yoruba (47,996 raw) retained under 30%, and Wolof (2,351 raw, the
  smallest Wikipedia source in this corpus) retained under 50% â
  article count alone does not predict data quality or usability.

## 8. Maintenance

This corpus was assembled for this project. Sources, cleaning code, and
spot-check reports are version-controlled alongside this data card. All
cleaning rules were applied identically and retroactively across every
Wikipedia-sourced language whenever a new rule was introduced, so
cross-language comparisons in this document reflect real data differences,
not inconsistent handling introduced partway through the project. Wolof's
supplementary-source procedure (Section 2.1) is the one documented
departure from this identical-pipeline approach, taken in direct response
to a corpus-size shortfall rather than as a default step for every
language.
