"""
Compares MAGNET's per-language boundary-predictor routing
(src/model/magnet.py's LanguageRoutedBoundaryPredictor, this project's own
extension) against the paper's script-level routing (the original
ScriptRoutedBoundaryPredictor design) on the fairness metric used
throughout this project: bytes/segment and its coefficient of variation
across the 9 languages.

Neither set of numbers is re-derived here — both are taken directly from
real eval_holdout inspection runs (inspect_segmentation.py):
- script_level: checkpoints_all_9langs/step_4999.pt, uniform
  beta_by_script=(0.5, 0.5) — same source as src/eval/bpe_baseline.py's
  MAGNET_STATS. Predates train.py's --seed fix, so this specific run isn't
  reproducible from current code (and can't be, since its architecture,
  ScriptRoutedBoundaryPredictor, was since removed) — kept as a historical
  reference point only.
- per_language: checkpoints_per_language/step_4999.pt, uniform
  beta_by_language=(0.5,)*9, seed=42 (train.py's default) — from a real,
  reproducible Colab run, pasted directly.

Read-only: loads no checkpoint, runs no model.

Usage:
    python -m src.eval.compare_routing_designs
"""
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "per_language_vs_script_level_comparison.json"

LANGUAGES = ("sw", "zu", "yo", "ig", "ha", "ny", "am", "rw", "wo")

SCRIPT_LEVEL_CHECKPOINT_INFO = {
    "checkpoint_dir": "checkpoints_all_9langs",
    "step": 4999,
    "beta_by_script": [0.5, 0.5],
}
PER_LANGUAGE_CHECKPOINT_INFO = {
    "checkpoint_dir": "checkpoints_per_language",
    "step": 4999,
    "beta_by_language": [0.5] * 9,
    "seed": 42,
}

# avg_bytes_per_segment from full eval_holdout inspect_segmentation.py runs
SCRIPT_LEVEL_BYTES_PER_SEGMENT = {
    "sw": 2.34, "zu": 2.25, "yo": 2.03, "ig": 2.30, "ha": 2.55,
    "ny": 2.35, "am": 1.96, "rw": 2.17, "wo": 2.29,
}
PER_LANGUAGE_BYTES_PER_SEGMENT = {
    "sw": 1.69, "zu": 2.21, "yo": 2.16, "ig": 2.31, "ha": 2.22,
    "ny": 2.30, "am": 2.17, "rw": 1.83, "wo": 1.81,
}


def coefficient_of_variation(values):
    """Population stdev / mean — lower = more consistent (fairer) across languages."""
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean


def main():
    cv_script_level = coefficient_of_variation([SCRIPT_LEVEL_BYTES_PER_SEGMENT[l] for l in LANGUAGES])
    cv_per_language = coefficient_of_variation([PER_LANGUAGE_BYTES_PER_SEGMENT[l] for l in LANGUAGES])

    # Amharic is the only Geez-script language in this project's 9, so it
    # already had a dedicated predictor under script-level routing —
    # identical setup to per-language routing, for Amharic specifically.
    # Isolating the 8 Latin-script languages (which went from sharing ONE
    # predictor to each having their own) removes that confound and shows
    # the routing change's actual effect.
    latin_languages = tuple(l for l in LANGUAGES if l != "am")
    cv_script_level_latin_only = coefficient_of_variation(
        [SCRIPT_LEVEL_BYTES_PER_SEGMENT[l] for l in latin_languages]
    )
    cv_per_language_latin_only = coefficient_of_variation(
        [PER_LANGUAGE_BYTES_PER_SEGMENT[l] for l in latin_languages]
    )

    comparison = [
        {
            "language": lang,
            "per_language_bytes_seg": PER_LANGUAGE_BYTES_PER_SEGMENT[lang],
            "script_level_bytes_seg": SCRIPT_LEVEL_BYTES_PER_SEGMENT[lang],
        }
        for lang in LANGUAGES
    ]

    results = {
        "script_level_checkpoint": SCRIPT_LEVEL_CHECKPOINT_INFO,
        "per_language_checkpoint": PER_LANGUAGE_CHECKPOINT_INFO,
        "comparison": comparison,
        "cv_per_language": cv_per_language,
        "cv_script_level": cv_script_level,
        "latin_only_note": (
            "am is the sole Geez-script language, so it had a dedicated "
            "predictor under script-level routing too — identical setup to "
            "per-language routing for am specifically. cv_*_latin_only "
            "isolates the 8 Latin-script languages, which went from sharing "
            "one predictor (script-level) to each having their own "
            "(per-language), removing that confound."
        ),
        "cv_per_language_latin_only": cv_per_language_latin_only,
        "cv_script_level_latin_only": cv_script_level_latin_only,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved: {RESULTS_PATH}")

    print(f"\n{'lang':<6}{'per_language':<15}{'script_level':<15}")
    for row in comparison:
        print(f"{row['language']:<6}{row['per_language_bytes_seg']:<15.4f}{row['script_level_bytes_seg']:<15.4f}")
    print("\ncoefficient of variation (fairness, lower = more equitable):")
    print(f"  per_language (all 9):        {cv_per_language:.4f}")
    print(f"  script_level (all 9):        {cv_script_level:.4f}")
    print(f"  per_language (8 Latin only): {cv_per_language_latin_only:.4f}")
    print(f"  script_level (8 Latin only): {cv_script_level_latin_only:.4f}")


if __name__ == "__main__":
    main()
