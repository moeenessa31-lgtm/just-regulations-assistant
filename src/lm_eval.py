"""Measurements shared by the training scripts: perplexity and greedy completions."""
import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mean_loss(model, dataset, batch_size: int = 8) -> float:
    """Average next-token loss over every token of the dataset."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        n = batch["input_ids"].numel()
        total += loss.item() * n
        count += n
    if was_training:
        model.train()
    return total / count


def perplexity(model, dataset, batch_size: int = 8) -> float:
    """exp(loss): roughly how many tokens the model is 'choosing between' at each step.
    Lower is better."""
    return math.exp(mean_loss(model, dataset, batch_size))


@torch.no_grad()
def complete(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    """Greedy (temperature 0) continuation of a prompt; returns only the new text."""
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
