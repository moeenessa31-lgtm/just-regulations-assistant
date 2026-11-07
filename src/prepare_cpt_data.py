"""Build the continued-pretraining corpus: clean every document and split by document.

Reads   data/text/*.txt
Writes  data/cpt/train.jsonl, data/cpt/heldout.jsonl  (one JSON line per document)
        data/cpt/stats.json                          (documents and characters per split)

The split is by whole document (never inside a document), so the held-out
perplexity measures text the model has never seen. The held-out documents are
listed in config.yaml.

Usage:  python -m src.prepare_cpt_data
"""
import json

from src.text_cleaning import clean_text
from src.utils import ROOT, load_config


def main() -> None:
    cfg = load_config()["cpt_data"]
    text_dir = ROOT / cfg["text_dir"]
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_names = set(cfg["heldout_docs"])
    excluded = set(cfg.get("exclude_docs") or [])
    splits = {"train": [], "heldout": []}

    for path in sorted(text_dir.glob("*.txt")):
        if path.stem in excluded:
            continue
        text = clean_text(path.read_text(encoding="utf-8"))
        if len(text) < cfg["min_chars"]:
            print(f"skip (too short): {path.stem}")
            continue
        split = "heldout" if path.stem in heldout_names else "train"
        splits[split].append({"doc": path.stem, "text": text})

    missing = heldout_names - {d["doc"] for d in splits["heldout"]}
    if missing:
        raise SystemExit(f"held-out documents not found: {sorted(missing)}")

    stats = {}
    for name, docs in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        stats[name] = {
            "documents": len(docs),
            "characters": sum(len(d["text"]) for d in docs),
            "doc_names": [d["doc"] for d in docs],
        }
    total = stats["train"]["characters"] + stats["heldout"]["characters"]
    stats["heldout_share_of_characters"] = round(stats["heldout"]["characters"] / total, 3)

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    for name in splits:
        print(f"{name}: {stats[name]['documents']} docs, {stats[name]['characters']:,} chars")
    print(f"held-out share: {stats['heldout_share_of_characters']:.1%}")


if __name__ == "__main__":
    main()
