import torch

from .dataset import LANGUAGES, PAD_ID

LANGUAGE_TO_ID = {lang: idx for idx, lang in enumerate(LANGUAGES)}


def collate_batch(examples, pad_id=PAD_ID):
    """
    Batches a list of [[dataset.MagnetByteDataset]] examples
    (`{"input_ids", "lang"}`) into right-padded tensors matching what
    [[hourglass_transformer.HourglassTransformer.forward]] /
    [[magnet.MAGNET.forward]] expect:

    - input_ids: (batch, max_len) long, right-padded with pad_id
    - attention_mask: (batch, max_len) bool, True at real (non-padded)
      positions
    - language_ids: (batch,) long, indexing into [[dataset.LANGUAGES]]
    """
    lengths = [ex["input_ids"].size(0) for ex in examples]
    batch_size = len(examples)
    max_len = max(lengths)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    language_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(examples):
        length = lengths[i]
        input_ids[i, :length] = ex["input_ids"]
        attention_mask[i, :length] = True
        language_ids[i] = LANGUAGE_TO_ID[ex["lang"]]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "language_ids": language_ids,
        "langs": [ex["lang"] for ex in examples],
    }
