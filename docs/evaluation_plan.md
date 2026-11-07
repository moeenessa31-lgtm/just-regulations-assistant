# Phase 2 — Evaluation plan

Goal: measure what each training stage changed, on the same 30 Phase 1 test
questions (`eval/seed_test_set.csv`), with the same settings for every model.

## 1. Models compared

| Version | What it is |
|---|---|
| Base | GPT-2 small (124M), no training on our data |
| + CPT | Base + continued pretraining on the cleaned JUST documents (`train_cpt.py`) |
| + SFT | CPT model + supervised fine-tuning on question–answer pairs (`train_sft.py`) |

## 2. Continued-pretraining measurements (Part A, step 1)

- **Held-out perplexity** before and after CPT, on 2 whole documents never seen in
  training (`student_conduct_code`, `disability_accommodations`, about 9.5% of the
  characters). They were chosen because no test question depends on them, so every
  document behind the test set stays in training.
- **Loss curves**: training loss every 5 steps, validation loss every 10 steps and
  after each pass (`results/cpt_loss_curve.png`).
- **Overfitting guard**: validation loss is checked after every pass; training stops
  when it does not fall, and the best pass is kept.
- **Same 5 prompts** completed before and after (`results/cpt_completions.md`).

## 3. Running the test questions (Part B)

- `score.py` calls all three models through one function, with the same chat
  template, **greedy decoding (temperature 0)** and the same maximum answer length
  (`generation` in `config.yaml`).
- The base model gets the question with the same template; it is expected not to
  follow the format. It is scored anyway as the starting point.
- No test question, or paraphrase of one, may appear in the SFT pairs (check in §5).

## 4. Scoring rules

| Answer type | Rule |
|---|---|
| Short exact answer (a number, a grade symbol, yes/no) | Exact match after normalizing case, spaces, Arabic letter variants (أ/إ/آ → ا, ى → ي, ة → ه) and Arabic-Indic digits (٧ → 7) |
| Longer answer | Graded by hand: **1** correct, **0.5** partly correct, **0** wrong |

An answer is **correct** only if it states the same rule and every number (weeks,
%, hours, GPA) as the expected answer and nothing that contradicts the documents
(same definition as the Phase 1 proposal).

**Two graders**: two team members grade every hand-graded answer independently,
then settle disagreements together. We report the **first-pass agreement rate**
(answers with the same score / answers graded).

**Reported per model**
- **Accuracy** = total points / 30.
- **Refusal accuracy** = out-of-scope questions (ids 28–30) answered with
  «لا أعرف» / "I don't know" ÷ 3. Reported beside accuracy, since a model that refuses
  everything would score 100% here.
- **Average latency** (seconds per answer).

## 5. Keeping SFT data apart from the test set

1. SFT pairs are written or generated from the documents, without looking at the test file.
2. Automatic check: every SFT question is compared with every test question
   (character n-gram overlap); pairs above a similarity threshold are removed.
3. Hand check of the flagged pairs and of at least 50 random pairs.

## 6. Outputs

- `results/<model>_answers.csv`: question, expected answer, model answer, score, note
  — one file per model; anyone on the team can re-run `python score.py`.
- `SCOREBOARD.md`: one row per model (held-out perplexity, accuracy, refusal
  accuracy, average latency, notes), extended in every later phase on the same test set.

## 7. Error analysis (SFT model)

Every wrong or partly correct SFT answer is put into one cause: format failure,
hallucination, knowledge missing, refusal error, or test-case problem (fixed in the
dev set only and noted in the decision log).
