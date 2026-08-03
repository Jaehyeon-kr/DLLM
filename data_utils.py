import torch

from tokenizer import get_tokenizer


class SFTCollator:
    """Pretraining packs documents to a fixed length so it never needs a collator. SFT samples
    are whole conversations of differing length, so pad them to the longest in the batch.

    query_mask marks the assistant turn, the only region finetune_sft.py is allowed to mask
    and take loss on. Padding gets 0 there, which keeps it out of both."""

    def __init__(self, hf_model_name):
        tokenizer = get_tokenizer(hf_model_name)

        ### tokenizer.py points pad at EOS, so padding is a real vocab id the model can attend ###
        ### to. Harmless here because query_mask keeps it unmasked and out of the loss ###
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = [torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
        query_mask = [torch.as_tensor(sample["query_mask"], dtype=torch.float) for sample in batch]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = torch.full((len(batch), max_length), self.pad_token_id, dtype=torch.long)
        padded_query_mask = torch.zeros((len(batch), max_length), dtype=torch.float)

        for i, (ids, mask) in enumerate(zip(input_ids, query_mask)):
            padded_input_ids[i, :len(ids)] = ids
            padded_query_mask[i, :len(mask)] = mask

        return {"input_ids": padded_input_ids, "query_mask": padded_query_mask}
