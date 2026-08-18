from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

PAD_ID = 256
VOCAB_SIZE = 257  # 256 raw byte values + 1 dedicated pad id

SCRIPTS = ("Latin", "Geez")

DEFAULT_LANGUAGES_YAML = Path(__file__).resolve().parents[2] / "configs" / "languages.yaml"

# Per-language folder names under DATA_ROOT. Train and eval are split out
# explicitly because Wolof's eval set was drawn from the Wikipedia-only
# corpus ("wo_wikipedia") while its train set is the combined
# Wikipedia + MasakhaNER corpus ("wo_combined") — see data_card.md
# Section 2.1. This is intentional, not a naming inconsistency to fix.
LANG_TO_FOLDER = {
    "am": {"train": "am_wikipedia", "eval": "am_wikipedia"},
    "ha": {"train": "ha_wikipedia", "eval": "ha_wikipedia"},
    "ig": {"train": "ig_wikipedia", "eval": "ig_wikipedia"},
    "ny": {"train": "ny_wikipedia", "eval": "ny_wikipedia"},
    "rw": {"train": "rw_wikipedia", "eval": "rw_wikipedia"},
    "sw": {"train": "sw_wikipedia", "eval": "sw_wikipedia"},
    "wo": {"train": "wo_combined", "eval": "wo_wikipedia"},
    "yo": {"train": "yo_wikipedia", "eval": "yo_wikipedia"},
    "zu": {"train": "zu_wikipedia", "eval": "zu_wikipedia"},
}


def _load_script_by_lang(languages_yaml_path):
    with open(languages_yaml_path, "r", encoding="utf-8") as f:
        languages = yaml.safe_load(f)["languages"]
    return {code: entry["script"] for code, entry in languages.items()}


def _corpus_path(data_root, lang, split):
    folder = LANG_TO_FOLDER[lang][split]
    if split == "train":
        return Path(data_root) / folder / "train" / "corpus.txt"
    return Path(data_root) / "eval_holdout" / folder / "eval.txt"


def _chunk_bytes(byte_ids, max_seq_len, stride=None):
    """
    Splits a flat sequence of byte ids into fixed-length chunks of
    max_seq_len (paper Appendix D.1), sliding by `stride` (default:
    max_seq_len, i.e. no overlap). A final remainder shorter than 2 bytes
    is dropped (next-token prediction needs at least one input/target pair).
    """
    stride = stride or max_seq_len
    n = len(byte_ids)
    chunks = []
    for start in range(0, n, stride):
        chunk = byte_ids[start:start + max_seq_len]
        if len(chunk) >= 2:
            chunks.append(chunk)
        if start + max_seq_len >= n:
            break
    return chunks


class MagnetByteDataset(Dataset):
    """
    Byte-level dataset for MAGNET (Ahia et al., 2024). Reads each language's
    train/corpus.txt (or eval_holdout/<folder>/eval.txt for the eval
    split — already the correct held-out set per data_card.md, never
    re-split or re-shuffled here), converts text to raw UTF-8 bytes, tags
    each resulting chunk with its language's script (configs/languages.yaml),
    and chunks into fixed max_seq_len windows for
    [[collate.collate_batch]] / [[magnet.MAGNET]].
    """

    def __init__(self, data_root, lang_codes, split="train", max_seq_len=512,
                 stride=None, languages_yaml_path=DEFAULT_LANGUAGES_YAML):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.stride = stride

        script_by_lang = _load_script_by_lang(languages_yaml_path)

        self.examples = []  # (byte_id_chunk: list[int], script: str, lang: str)
        for lang in lang_codes:
            corpus_path = _corpus_path(self.data_root, lang, split)
            text = corpus_path.read_text(encoding="utf-8")
            byte_ids = list(text.encode("utf-8"))
            script = script_by_lang[lang]
            for chunk in _chunk_bytes(byte_ids, max_seq_len, stride):
                self.examples.append((chunk, script, lang))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk, script, lang = self.examples[idx]
        return {
            "input_ids": torch.tensor(chunk, dtype=torch.long),
            "script": script,
            "lang": lang,
        }
