"""Turn the cleaned documents into fixed-length token blocks for next-token training.

Each document is tokenized, an end-of-text token is added after it, all documents
are joined into one long stream, and the stream is cut into blocks of
`block_size` tokens. The leftover tokens at the end (less than one block) are dropped.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_docs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_docs(docs: list[dict], tokenizer) -> list[int]:
    stream = []
    for d in docs:
        stream.extend(tokenizer(d["text"])["input_ids"])
        stream.append(tokenizer.eos_token_id)
    return stream


class BlockDataset(Dataset):
    """Blocks of `block_size` tokens; the labels are the inputs (the model shifts them)."""

    def __init__(self, token_ids: list[int], block_size: int):
        n_blocks = len(token_ids) // block_size
        if n_blocks == 0:
            raise ValueError(f"only {len(token_ids)} tokens, fewer than one block of {block_size}")
        ids = torch.tensor(token_ids[: n_blocks * block_size]).view(n_blocks, block_size)
        self.blocks = ids

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict:
        x = self.blocks[i]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}
