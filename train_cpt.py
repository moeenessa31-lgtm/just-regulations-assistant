"""Stage 1 — Continued pretraining (CPT).

Keeps training the base model with the next-token objective on our cleaned JUST
documents, so it picks up the vocabulary and facts of the regulations.

Steps:
  1. Load the base model and the train / held-out documents (data/cpt/).
  2. BEFORE training: held-out perplexity + completions of the 5 sample prompts.
  3. Train for up to `num_train_epochs` passes; a checkpoint is saved after each
     pass. Validation loss is measured every few steps for the curve, and after
     each pass for early stopping: if it stops falling, training stops and the
     best checkpoint is kept (overfitting guard).
  4. AFTER training: the same measurements on the best checkpoint.
  5. Save results/cpt_metrics.json, results/cpt_loss_curve.png,
     results/cpt_completions.md and the final model in outputs/cpt/final.

Usage:
  python -m src.prepare_cpt_data        # once, builds data/cpt/
  python train_cpt.py --smoke           # 5% of data, ~20 steps: check the pipeline
  python train_cpt.py                   # full run (GPU: Kaggle / Colab)
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.cpt_dataset import BlockDataset, read_docs, tokenize_docs
from src.lm_eval import complete, mean_loss, perplexity
from src.utils import ROOT, load_config, set_seed


class ValLossCurve(TrainerCallback):
    """Measures held-out loss every N steps, only to draw a smooth validation curve."""

    def __init__(self, dataset, every: int, batch_size: int):
        self.dataset, self.every, self.batch_size = dataset, every, batch_size
        self.points = []  # (step, epoch, loss)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0:
            loss = mean_loss(model, self.dataset, self.batch_size)
            self.points.append((state.global_step, state.epoch, loss))


def run_prompts(model, tokenizer, prompts, max_new_tokens):
    return [{"prompt": p, "completion": complete(model, tokenizer, p, max_new_tokens)} for p in prompts]


def plot_curves(log_history, val_points, path: Path):
    train = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    plt.figure(figsize=(7, 4))
    if train:
        plt.plot(*zip(*train), label="training loss")
    if val_points:
        plt.plot([p[0] for p in val_points], [p[2] for p in val_points], "o-", label="validation loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Continued pretraining")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--model", help="override base_model from config.yaml (e.g. a local folder)")
    args = ap.parse_args()

    cfg = load_config()
    c = cfg["cpt"]
    set_seed(cfg["seed"])
    model_name = args.model or cfg["base_model"]
    out_dir = ROOT / (c["output_dir"] + ("_smoke" if args.smoke else ""))
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    tag = "cpt_smoke" if args.smoke else "cpt"

    # 1. model, tokenizer, data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.to("cuda")

    data_dir = ROOT / cfg["cpt_data"]["out_dir"]
    train_docs = read_docs(data_dir / "train.jsonl")
    heldout_docs = read_docs(data_dir / "heldout.jsonl")
    train_ids = tokenize_docs(train_docs, tokenizer)
    heldout_ids = tokenize_docs(heldout_docs, tokenizer)
    n_chars = sum(len(d["text"]) for d in train_docs + heldout_docs)
    tokens_per_char = (len(train_ids) + len(heldout_ids)) / n_chars
    corpus_tokens = {"train": len(train_ids), "heldout": len(heldout_ids)}
    if args.smoke:
        train_ids = train_ids[: max(c["block_size"] * 4, int(len(train_ids) * c["smoke_fraction"]))]
    train_ds = BlockDataset(train_ids, c["block_size"])
    heldout_ds = BlockDataset(heldout_ids, c["block_size"])
    print(f"corpus: {corpus_tokens['train']:,} train + {corpus_tokens['heldout']:,} held-out tokens "
          f"({tokens_per_char:.2f} tokens per character) | used: {len(train_ds)} train blocks, "
          f"{len(heldout_ds)} held-out blocks")

    # 2. before training
    bs = c["batch_size"]
    gen = cfg["generation"]
    ppl_before = perplexity(model, heldout_ds, bs)
    print(f"held-out perplexity BEFORE: {ppl_before:.2f}")
    before = run_prompts(model, tokenizer, cfg["sample_prompts"], gen["max_new_tokens"])

    # 3. training
    strategy = "steps" if args.smoke else "epoch"
    steps_per_epoch = -(-len(train_ds) // (bs * c["gradient_accumulation_steps"]))
    total_steps = c["smoke_max_steps"] if args.smoke else steps_per_epoch * c["num_train_epochs"]
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=c["num_train_epochs"],
        max_steps=c["smoke_max_steps"] if args.smoke else -1,
        learning_rate=float(c["learning_rate"]),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        warmup_steps=int(c["warmup_ratio"] * total_steps),
        weight_decay=c["weight_decay"],
        eval_strategy=strategy,
        save_strategy=strategy,              # a checkpoint after each pass
        eval_steps=10,
        save_steps=10,
        save_only_model=True,                # no optimizer state: smaller checkpoints
        load_best_model_at_end=True,         # keep the pass with the lowest validation loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    curve = ValLossCurve(heldout_ds, c["val_eval_every_steps"], bs)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=heldout_ds,
        callbacks=[curve, EarlyStoppingCallback(early_stopping_patience=c["early_stopping_patience"])],
    )
    trainer.train()
    model = trainer.model

    # 4. after training
    ppl_after = perplexity(model, heldout_ds, bs)
    print(f"held-out perplexity AFTER:  {ppl_after:.2f}")
    after = run_prompts(model, tokenizer, cfg["sample_prompts"], gen["max_new_tokens"])

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # 5. save results
    epoch_evals = [{"epoch": h["epoch"], "eval_loss": h["eval_loss"]}
                   for h in trainer.state.log_history if "eval_loss" in h]
    metrics = {
        "base_model": model_name,
        "settings": {k: c[k] for k in ("block_size", "num_train_epochs", "learning_rate",
                                       "batch_size", "early_stopping_patience")},
        "corpus_tokens_train": corpus_tokens["train"],
        "corpus_tokens_heldout": corpus_tokens["heldout"],
        "train_blocks": len(train_ds),
        "heldout_blocks": len(heldout_ds),
        "tokens_per_character": round(tokens_per_char, 3),
        "heldout_perplexity_before": round(ppl_before, 3),
        "heldout_perplexity_after": round(ppl_after, 3),
        "epochs_completed": round(trainer.state.epoch, 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "eval_loss_per_pass": epoch_evals,
        "val_curve": [{"step": s, "epoch": round(e, 3), "loss": round(l, 4)} for s, e, l in curve.points],
    }
    with open(results_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_curves(trainer.state.log_history, curve.points, results_dir / f"{tag}_loss_curve.png")
    with open(results_dir / f"{tag}_completions.md", "w", encoding="utf-8") as f:
        f.write("# Sample completions before and after continued pretraining\n\n")
        f.write(f"Greedy decoding, max {gen['max_new_tokens']} new tokens.\n\n")
        for b, a in zip(before, after):
            f.write(f"## Prompt: {b['prompt']}\n\n**Before:** {b['completion']}\n\n"
                    f"**After:** {a['completion']}\n\n")
    print(f"saved results/{tag}_*.json/png/md and model in {final_dir}")

    # optional: push the model to the Hugging Face Hub (checkpoints never go to Git)
    if c.get("hub_repo") and not args.smoke and os.environ.get("HF_TOKEN"):
        model.push_to_hub(c["hub_repo"], private=True, token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(c["hub_repo"], private=True, token=os.environ["HF_TOKEN"])
        print(f"pushed to https://huggingface.co/{c['hub_repo']}")


if __name__ == "__main__":
    main()
