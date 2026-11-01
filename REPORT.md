# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `sopa` |
| Members | Joceline Nicole Ayala Alemán; Eduardo Noé Figueroa Rodríguez; Ever Alejandro Montes Quintanilla; Oscar Enrique Peréz Osorio |
| Repository URL | https://github.com/JocelineAyala/sopa-qa |
| `submission` commit hash | **PENDING:** create/verify the `submission` tag and record its commit hash. |
| Public service URL used at the dry run | **PENDING:** official instructor dry-run URL and confirmation. Internal HTTPS tests are not that dry run. |
| AI tools used (which, and for what) | ChatGPT and Codex: programming, diagnosis, and documentation. |

## 1. Model choice and justification

We trained `FacebookAI/roberta-base` on SQuAD 2.0 and serve **V2** on CPU. We considered DistilBERT for its smaller size and CPU cost, but did not train or benchmark it.

Our separate experiments were V1/V2 full fine-tuning, V3 continued training, V4 LoRA, and V5/V6 error-focused training. V1's evaluation EM/F1 was 77.0688/80.5217. V3/V4 candidates lowered answerable EM; V6 lowered unanswerable EM, so all kept V2. V5 improved monitoring results but not balanced performance on 9,873 questions:

| Model evaluated | Overall EM/F1 | Answerable EM/F1 | Unanswerable EM/F1 |
|---|---:|---:|---:|
| V2 | 79.0540 / 82.2519 | 74.1564 / 80.5361 | 83.9764 / 83.9764 |
| V5 candidate | 79.1249 / 82.0755 | 73.4088 / 79.2951 | 84.8700 / 84.8700 |

Sources: [V1 metrics](logs/roberta_squad2/metricas.json), [V2 metrics](logs/roberta_squad2_v2/metricas.json), experiment `resumen.json` files, and [V5 evaluation](logs/roberta_squad2_v5_rapido/metricas_evaluacion.json).

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full train: 130,319 examples; 131,821 tokenized windows |
| Validation data (what you validated on, how many examples) | First 2,000 validation examples after shuffle(seed=42); 2,042 windows. Also used for V2 threshold calibration. Remaining 9,873 for subsequent evaluation. |
| Epochs | Maximum 2; stopped at 1.6386 epochs / 13,500 optimizer steps; selected checkpoint-10500 |
| Learning rate and schedule (warmup?) | 1e-5; linear; 824 warmup steps; AdamW; weight decay 0.01 |
| Batch size | Train/evaluation 2; gradient accumulation 8; effective training batch 16 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | RTX 4060 Laptop 8 GB; FP16 and gradient checkpointing; recorded Trainer runtime 7,682.1662 s (128.04 min) |

We evaluated/saved every 500 steps with early-stopping patience 6. Our best validation loss was 0.835745; mean training loss was 1.102855. Evidence: [script](versiones/v2/entrenar_v2.py), [configuration](logs/roberta_squad2_v2/configuracion.json), [summary](logs/roberta_squad2_v2/resumen.json), and [loss curve](logs/roberta_squad2_v2/loss.png).

Calibrated validation results from V2 metrics:

| Group | Count | EM (%) | F1 (%) |
|---|---:|---:|---:|
| Overall | 2,000 | 79.8500 | 82.5035 |
| Answerable | 979 | 74.5659 | 79.9867 |
| Unanswerable | 1,021 | 84.9167 | 84.9167 |

We reused development data. Later experiments split the same 2,000 questions into 1,000 calibration and 1,000 monitoring examples. We also repeatedly evaluated the other 9,873; neither partition is an independent test.

## 3. No-answer threshold

We compute `difference = minimum CLS start+end score across windows − best valid span score`. We return `answer=null, start=null` when `difference > -0.5847749710083008`. We chose this threshold by maximizing F1, then EM, on 2,000 questions. We load [umbral.json](logs/roberta_squad2_v2/umbral.json), not historical alternative-threshold files.

A lower threshold increases abstention and missed answerable questions; a higher threshold increases unsupported answers. `score = sigmoid(min(abs(difference − threshold), 60))` expresses distance from the decision boundary. **It is not a calibrated probability of correctness.**

## 4. Serving architecture

We use FastAPI/Uvicorn in [app.py](app.py), with one worker. Startup requires a token, loads local V2 weights/tokenizer in CPU float32, checks settings, and warms up inference. `/health` returns 503 while loading and 200 when ready.

We validate `/predict` JSON and run inference in a thread with a lock. [qa_v3.py](qa_v3.py) uses overlapping windows, top-20 start/end positions, and a 30-token answer limit, returning a context substring and offset. We check one Bearer header against `QA_API_TOKEN` on `/info` and `/predict` using constant-time comparison.

We distribute weights/tokenizer in [release modelo-v2](https://github.com/JocelineAyala/sopa-qa/releases/tag/modelo-v2), asset `sopa-qa-v2-modelo.zip`. It contains `modelo/`: extract into `outputs/roberta_squad2_v2/`. PowerShell, with Python, Git, and ngrok installed:

```powershell
git clone https://github.com/JocelineAyala/sopa-qa.git
cd sopa-qa
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch transformers safetensors numpy datasets tqdm fastapi "pydantic>=2" uvicorn
Invoke-WebRequest https://github.com/JocelineAyala/sopa-qa/releases/download/modelo-v2/sopa-qa-v2-modelo.zip -OutFile sopa-qa-v2-modelo.zip
Expand-Archive sopa-qa-v2-modelo.zip -DestinationPath outputs/roberta_squad2_v2
if (!(Test-Path outputs/roberta_squad2_v2/modelo/config.json)) { throw "Check ZIP layout: place model files at the required path" }
if (!(Test-Path logs/roberta_squad2_v2/umbral.json)) { throw "Missing V2 threshold" }
$env:QA_API_TOKEN = Read-Host "Service token"
$env:QA_TEAM = "sopa"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
# Separate terminal
ngrok http 8000
```

On November 1, 2026, we verified a clean copy with Python 3.14.0, a new environment, and downloaded weights; we corrected extraction and removed the GitHub CLI requirement. [Repeat steps](verificaciones/COPIA_LIMPIA.md) and [package versions](verificaciones/copia_limpia_requirements.txt) are recorded. Our [ngrok backup test](https://metaphorically-spoollike-wan.ngrok-free.dev) was not an official dry run; availability is temporary.

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---:|---|
| sample-01 | `16` / `16 attention heads` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3 times` | 310 | No |
| sample-05 | `50,257` / `50,257 tokens` | `50,257` | 3157 | Yes |

We recorded valid contracts 5/5 and reference matches 4/5 in [local](logs/roberta_squad2/api_local.json), [ngrok](logs/roberta_squad2/api_tunel.json), and [clean-copy tests](verificaciones/copia_limpia.json). The tunnel log now contains ngrok results, not Cloudflare. Clean-copy health/info, authentication, and invalid-request checks passed; we stopped our temporary server.

Sample-04 still fails: we return GPT-3's memory-reduction figure for BERT-base, with score 0.633084. We have no verified fix or instructor test result.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 124,056,578 |
| Size on disk | Exported model: 476.7 MiB (team-provided measurement) |
| Mean latency per `/predict` request on your laptop (CPU) | Earlier local HTTP: 237.69 ms; ngrok HTTPS: 1,051.57 ms; clean-copy HTTP on port 8001: 322.92 ms. Five requests per run, including sample-05; not a load benchmark. |
| Laptop CPU and RAM | AMD Ryzen 7 8845HS; 15.3 GB usable RAM |

We obtained parameters from [verification](verificaciones/resultado.json) and averaged five request times per run, including overhead. We did not double-count `api_ngrok.json`, which duplicates the tunnel log.

## 7. Problems encountered and how you fixed them

1. **Problem/cause:** our initial `probar_entrenamiento.py` (`6ab0fc3`) used only 256 examples and 20 steps, without validation or best-model selection. **Fix:** we added full training, 2,000 validation examples, checkpoint selection, and resume support in **`3cc507b`** (`entrenar.py`). This was a workflow gap, not a crash.
2. **Problem/cause:** relative paths depended on the working directory. **Fix:** we added `project_paths.ROOT`, package imports, and direct-execution setup in **`13b9c45`**. [Verification](verificaciones/resultado.json) confirms imports from another directory; no undocumented failure incident is claimed.

## 8. Decoder investigation

**What could a decoder or encoder–decoder do differently from our RoBERTa?**

Our RoBERTa returns one continuous context span. A GPT-style decoder generates tokens; an encoder–decoder processes input before generating an answer. Generation allows yes/no answers and combinations of information from different passages (Radford et al., 2018; Vaswani et al., 2017; Raffel et al., 2020).

**How does this relate to our model's errors?**

In sample-04, a generated explanation could highlight the BERT-base/GPT-3 mismatch. However, generation alone guarantees neither correct reasoning nor recognition of missing answers.

**What problems and costs did we identify?**

Generative models can invent plausible answers unsupported by the source (Maynez et al., 2020, on summarization). For example, "The satellite was launched in 2024" does not identify its manufacturer; answering "NASA" would be unsupported. This illustrates the risk; we did not run this test.

A generated answer may lack a valid start offset, breaking our API contract. Successive generation steps can increase CPU latency; size and memory depend on the checkpoint.

**What measurements are needed to compare these alternatives?**

We would compare accuracy, abstention, context support, size, memory, and response time against the five-second limit. We used published research and error analysis; we did not implement or benchmark these alternatives.

Radford et al. (2018). [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

Vaswani et al. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762).

Raffel et al. (2020). [Exploring the limits of transfer learning with a unified text-to-text transformer](https://jmlr.org/papers/v21/20-074.html).

Maynez et al. (2020). [On faithfulness and factuality in abstractive summarization](https://doi.org/10.18653/v1/2020.acl-main.173).

## 9. Known limitations

We support English extractive QA, without retrieval, synthesis, or verified multilingual support. We limit context/question/ID lengths to 5,000/500/64 characters; invalid requests return 422, and missing/wrong authentication returns 401. We can return plausible false spans for unsupported premises. We serialize CPU requests and must keep tunnels available. Our calibration may not transfer to new data. The clean-copy check used the same laptop, not another OS or machine. **Pending:** official dry-run URL/evidence and `submission` tag/hash.
