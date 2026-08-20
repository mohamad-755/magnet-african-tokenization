# data/

The actual training/eval corpora for all 9 languages live in an external
Google Drive folder (`DATA_ROOT`), not here — point the pipeline at it via
`--data-root` or the `MAGNET_DATA_ROOT` env var. See `data_card.md` for full
corpus documentation.

`DATA_ROOT` (shared, view access): https://drive.google.com/drive/folders/1jN8eXuBZ5IxXoGHM-zPz9im6IajAd8ca
— also holds every experiment's checkpoints (`checkpoints_*/`), including
the ones this project's results are drawn from
(`checkpoints_all_9langs/`, `checkpoints_per_language/`,
`checkpoints_per_language_tuned_beta/`, `checkpoints_per_language_tuned_beta_lowreg/`).

Expected `DATA_ROOT` structure (only the parts the pipeline actually reads):

```
DATA_ROOT/
├── <lang>_wikipedia/train/corpus.txt   (wo_combined/ for Wolof's train split)
└── eval_holdout/<lang>_wikipedia/eval.txt
```

Each `<lang>_wikipedia/` folder also has a `cleaned/corpus.txt` +
`cleaning_report.md` and a `spot_check_report_<lang>.md` — outputs of the
cleaning pipeline documented in `data_card.md`. The pipeline here doesn't
read them; `train/corpus.txt` is already the cleaned, eval-holdout-excluded
split.

`raw/` is gitignored and currently empty. It previously held a local copy of
Hausa's 252MB corpus, worked around a Google Drive read failure hit on a
local machine (never in Colab, where actual training runs). If a language's
corpus fails to read reliably from Drive again, copying it here as
`raw/<folder>/train/corpus.txt` is the fix that worked before.
