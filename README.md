# JUST Regulations Assistant

A smart assistant that answers Data Science students' questions about the rules of
Jordan University of Science and Technology (JUST): absence, withdrawal, excuses,
exams, grading, graduation and course prerequisites. Every answer cites its source
document and article. If a question needs a personal record or an administrative
decision, the assistant says "I don't know".

Course: *Building LLM Systems: From Models to Production* — Team Project.
Current status: **Phase 2 (Train a Small Model and Evaluate It)**.

## Team (Phase 1 roles)

| Member | Role |
|---|---|
| Hamza Moeen Rasheed Issa | Product lead + Evaluation lead |
| Mohammad Ziad Abdelmajeed Alrousan | Data lead |
| Asem Zakaria Alfreeh | Engineering lead |

Roles rotate at the start of each phase.

## Repository layout

```
docs/       proposal.pdf, data_card.pdf (Phase 1 submission)
data/raw/   the 20 source PDFs and the study plan chart (see data/SOURCES.md)
data/samples/  5 sample files referenced by the data card
data/text/  text extracted from data/raw (OCR for scanned files; check numbers by hand)
eval/       seed_test_set.csv: 30 questions with expected answers (written before any model)
src/        extract_text.py
```

## Extract the text

```
python3 src/extract_text.py data/raw data/text
```

Needs `poppler-utils` and `tesseract` with the Arabic language pack (`tesseract-ocr-ara`).
Scanned PDFs are read with OCR, which can make mistakes in numbers.
`data/text/misc_academic_decisions_garbled.txt` comes from OCR because the PDF has broken fonts.

## Planned architecture (first version)

question -> retrieve top chunks from the documents (embeddings + source metadata)
-> LLM (free Hugging Face / NVIDIA API) -> answer with document and article citation.
Later phases: prerequisite table + an agent tool (RAG + agent), compared against the simpler RAG.

## Test set

`eval/seed_test_set.csv` has columns `id,question,expected_answer,source,difficulty`.
15 easy, 10 medium, 5 hard; questions 28-30 are out-of-scope and must be refused.
The file is UTF-8 with BOM so Excel shows the Arabic correctly.

## Phase 2: train a small model

Setup (Python 3.10+; a GPU is needed for the full runs, e.g. a free Kaggle T4):

```
pip install -r requirements.txt
```

All settings and the random seed are in `config.yaml`.

**Stage 1 — continued pretraining**

```
python -m src.prepare_cpt_data   # clean data/text, split 90/10 by document -> data/cpt/
python train_cpt.py --smoke      # 5% of the data, 20 steps: checks the pipeline
python train_cpt.py              # full run
```

Writes `results/cpt_metrics.json` (corpus tokens, held-out perplexity before/after),
`results/cpt_loss_curve.png`, `results/cpt_completions.md` (same 5 prompts before/after)
and the model in `outputs/cpt/final`. On Kaggle, run `notebooks/run_cpt_kaggle.ipynb`.

The evaluation plan is in `docs/evaluation_plan.md`.

**Checkpoints** are not in Git. Location: _TODO — Hugging Face Hub repo or Google Drive link_.
To push to the Hub, set `cpt.hub_repo` in `config.yaml` and the `HF_TOKEN`
environment variable (see `.env.example`).
