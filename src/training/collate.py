import torch

from .dataset import PAD_ID, SCRIPTS

SCRIPT_TO_ID = {script: idx for idx, script in enumerate(SCRIPTS)}


def collate_batch(examples, pad_id=PAD_ID):
    """
    Batches a list of [[dataset.MagnetByteDataset]] examples
    (`{"input_ids", "script", "lang"}`) into right-padded tensors matching
    what [[hourglass_transformer.HourglassTransformer.forward]] /
    [[magnet.MAGNET.forward]] expect:

    - input_ids: (batch, max_len) long, right-padded with pad_id
    - attention_mask: (batch, max_len) bool, True at real (non-padded)
      positions
    - script_ids: (batch,) long, indexing into [[dataset.SCRIPTS]]
    """
    lengths = [ex["input_ids"].size(0) for ex in examples]
    batch_size = len(examples)
    max_len = max(lengths)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    script_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(examples):
        length = lengths[i]
        input_ids[i, :length] = ex["input_ids"]
        attention_mask[i, :length] = True
        script_ids[i] = SCRIPT_TO_ID[ex["script"]]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "script_ids": script_ids,
        "langs": [ex["lang"] for ex in examples],
    }
