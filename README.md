# MLflow Champion-Challenger: Sentiment Classification

Student: Aarush Gandotra | USN: 1MS25CI001

## 1. Task and models
- **Task:** binary sentiment classification (POSITIVE / NEGATIVE) of single sentences.
- **Dataset:** SST-2 validation set (GLUE), a fixed random sample of **200 sentences** (seed 42), saved in `data/eval_inputs.csv`.
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (about 67M parameters).
- **Model B (Challenger):** `textattack/bert-base-uncased-SST-2` (about 110M parameters). It outputs `LABEL_0` / `LABEL_1`, which are mapped to NEGATIVE / POSITIVE inside the MLflow wrapper (`hf_models.py`).
- Both models run on the same inputs, in the same order, on CPU.

## 2. How to run
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# terminal 1 (keep it open)
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# terminal 2
python prepare_data.py          # builds data/eval_inputs.csv
python check_models.py          # sanity check of the label mapping
python evaluate.py              # logs both models, decides the Champion
python register_champion.py     # registers the models, sets the aliases
python predict_from_mlflow.py   # loads models:/sentiment-model@champion, predicts data/new_data.csv
```
MLflow UI: http://127.0.0.1:5000

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 on the evaluation set. **Higher is better.** (Accuracy, macro-precision and macro-recall: higher is better. `avg_latency_ms` and `total_inference_s`: lower is better.)
- **Minimum improvement margin:** the Challenger must beat the Champion by at least **0.01 (1 percentage point)** in macro-F1. <Explain why: with 200 sentences, one sentence is 0.5 percentage points, so a smaller difference could be noise.>
- **Tie-breaker:** if the difference is within the margin (between -0.01 and +0.01), the **faster model** (lower average latency) is chosen.
- **Guardrail:** a Challenger whose average latency is more than **5x** the Champion's is rejected.
- If the Challenger is lower by more than the margin, the Champion stays.

## 4. Results
| Metric | A: distilbert-sst2 | B: bert-sst2 | B - A |
|---|---|---|---|
| accuracy | <fill> | <fill> | <fill> |
| macro_f1 | <fill> | <fill> | <fill> |
| macro_precision | <fill> | <fill> | <fill> |
| macro_recall | <fill> | <fill> | <fill> |
| avg_latency_ms | <fill> | <fill> | <fill> |
| total_inference_s | <fill> | <fill> | <fill> |

Copy the numbers from the table printed by `evaluate.py` (also in `outputs/metrics_summary.json`).

**Examples where the models disagree** (from `outputs/disagreements.csv`):
| Sentence | Reference | A | B | My comment |
|---|---|---|---|---|
| <fill> | <fill> | <fill> | <fill> | <fill> |

## 5. Decision (150-250 words)
<Write it yourself after the run. Answer: Who is the Champion? Does the Challenger improve on the Champion, and by how much? Is the difference meaningful with 200 samples? What are the trade-offs (speed, size, the kind of errors)?>

## 6. Limitations and declaration
- **Sample size:** 200 sentences from one dataset; both models were fine-tuned on SST-2 training data, so the scores are probably higher than on other text.
- <Add anything else that may make the results unreliable.>
- **Tools used:** <State honestly what you used. Example: "An AI assistant helped me plan the project structure and draft the scripts. I completely understand the project I've created. I ran every script, checked the outputs and can explain each line.">