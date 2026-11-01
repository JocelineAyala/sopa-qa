# Clean-copy serving verification

We completed this check on November 1, 2026. It is a local reproduction check, not an instructor dry run or a test on another laptop/OS.

## Inputs and isolation

- Source commit: `13b9c45ed6043f117aa89f4015c5faa3a9c8aed3`, using the current working-tree contents of tracked files plus the then-current `REPORT.md`.
- We copied 121 files into a new system temporary directory outside the repository. Its path and the copied-file SHA-256 manifest are in [copia_limpia.json](copia_limpia.json).
- We excluded ignored files and `.venv/`, `data/`, `models/`, `outputs/`, and `tools/`. We created a new virtual environment with Python 3.14.0 and `include-system-site-packages = false`.
- We installed the report's serving package list from PyPI. Pip reused cached package wheels where available; it did not reuse the project virtual environment. `pip check` returned `No broken requirements found.`
- Full installed package versions: [copia_limpia_requirements.txt](copia_limpia_requirements.txt). The new environment used `torch==2.15.0` and `transformers==5.22.1`. No changes were made to the original environment.

## Release download and instruction fixes

We downloaded the public release asset directly, without GitHub CLI:

```powershell
Invoke-WebRequest https://github.com/JocelineAyala/sopa-qa/releases/download/modelo-v2/sopa-qa-v2-modelo.zip -OutFile sopa-qa-v2-modelo.zip
Expand-Archive sopa-qa-v2-modelo.zip -DestinationPath outputs/roberta_squad2_v2
```

The download was 457,206,335 bytes. ZIP SHA-256:

```text
e5a0eacd15ce2f93db653325188ab098fcce391b915de8be91e3db44d7be36d3
```

The archive contains `modelo/config.json`, `modelo/model.safetensors`, `modelo/tokenizer.json`, `modelo/tokenizer_config.json`, and `modelo/training_args.bin`. The report previously extracted into the repository root, which put `modelo/` in the wrong place. We corrected the temporary layout and the report: extract into `outputs/roberta_squad2_v2/`, so `outputs/roberta_squad2_v2/modelo/config.json` exists. The threshold comes from the copied Git file `logs/roberta_squad2_v2/umbral.json`.

GitHub CLI was absent on this machine. Direct public download succeeded, so the report no longer requires `gh`. No serving dependency was missing.

## API checks and actual results

We started one Uvicorn worker on `127.0.0.1:8001` with the new environment, local downloaded weights, offline Hugging Face mode, and CPU serving. A random token was generated in memory and passed only through the child process environment. We did not print or save it. No tunnel was started.

| Check | Result |
|---|---|
| `/health` after model loading | HTTP 200, `status: ok` |
| `/info` and `/predict`, missing token | HTTP 401 for both |
| `/info` and `/predict`, wrong token | HTTP 401 for both |
| Authenticated `/info` | `team: sopa`, `model: FacebookAI/roberta-base`, 124,056,578 parameters, length 384, stride 128 |
| Incomplete authenticated `/predict` request | HTTP 422, `bad_request` |
| Public-sample response contract | 5/5 valid; types, score bounds, null pairing, and context offsets checked |
| Public-sample reference matches | 4/5 |
| Mean request time, five public samples | 322.92 ms, local HTTP on port 8001; not a benchmark |

| Sample | Answer | Start | Reference match |
|---|---|---:|---|
| sample-01 | `16` | 383 | Yes |
| sample-02 | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | Yes |
| sample-04 | `3 times` | 310 | No; expected `null` |
| sample-05 | `50,257` | 3157 | Yes |

The detailed responses and timings are in [copia_limpia.json](copia_limpia.json). We stopped only the child server PID saved by this run; `temporary_server_stopped` is `true`. We did not train, alter original model files, stop existing servers/tunnels, commit, or push. Temporary files remain available for inspection.

## Repeat the check

The orchestration script runs from the repository and writes a new isolated copy. It needs permission to write the system temporary directory, download packages/weights, and start its own local server. Port 8001 must be free; it aborts if occupied.

```powershell
python verificaciones/verificar_copia_limpia.py --phase prepare
python verificaciones/verificar_copia_limpia.py --phase install
python verificaciones/verificar_copia_limpia.py --phase download
python verificaciones/verificar_copia_limpia.py --phase test
```

Run phases sequentially. `prepare` copies only tracked, nonignored project files plus `REPORT.md`; newly added verification files will be copied only after inclusion in Git. The script itself executes from the source repository. `test` checks dependencies and HTTP behavior, uses an in-memory token, and stops its server in a `finally` block. Each phase updates `verificaciones/copia_limpia.json`; preserve the evidence first if running again. To reproduce the exact tested environment, install the saved requirements rather than the unpinned package list.

The report instructions are corrected, but the official instructor dry run and the final `submission` tag/hash remain pending. This check does not establish portability to other operating systems or Python versions.
