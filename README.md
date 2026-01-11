# MASC Reproduction (Who&When)

## Data layout
- Place data under `./Who&When/Algorithm-Generated/*.json` and `./Who&When/Hand-Crafted/*.json`.
- Parsed trajectories are built from `question`, `history`, `mistake_agent`, `mistake_step`.

## Quick commands
```bash
python train.py --mode audit
python train.py --mode unsup --norm_mode prefix_only
python train.py --mode weak --ablate all
python train_hc.py --mode unsup --norm_mode prefix_only
python train_hc.py --mode weak --ablate all
python train.py --mode unsup --llm_stub
python train.py --mode weak --llm_stub --max_train_trajs 20 --max_eval_trajs 20
python train.py --mode unsup --norm_mode none
```

## Output locations
- Audit: `./outputs/audit_*.json`
- Unsupervised: `./outputs/unsup_*.json`
- Weak supervision: `./outputs/weak_*.json`
- Ablations: `./outputs/ablation_*.json`

## Notes on settings
- Unsupervised mode never uses `mistake_idx` as labels. If no true normal trajectories exist, it uses pre-mistake prefixes as pseudo-normal. If that is still empty and `use_pseudo_normal_training` is enabled, it generates pseudo-normal steps by LLM prompting (cached in `./.pseudo_normal_cache.jsonl`).
- Evaluation always truncates to the mistake step (truncate=False was removed) to avoid post-mistake leakage.
- `--norm_mode` controls scoring normalization: `prefix_only` uses running stats up to the current step, `full` uses stats over the evaluated prefix, and `none` disables normalization.
- Mapping methods:
  - `A_agent_local`: mistake_step is per-agent index.
  - `B_agent_global`: mistake_step is global agent step index.
  - `C_raw_to_agent`: mistake_step is raw history index mapped to nearest agent step.

## Common issues
- High `mistake_idx=-1` ratio: mapping failed or the file only has raw indices that do not align with agent steps. Run `python train.py --mode audit` and inspect `fail_samples`.
- No `is_correct` or `is_corrected` true: unsupervised mode will switch to prefix mining or pseudo-normal generation.
- Model download blocked: set `--allow_download` only if network access is available; otherwise keep cached models and `local_files_only`.
- If running on CPU, use `--llm_stub` for a lightweight debug backbone; disable it for paper-aligned runs on GPU.

## Project structure
- `masc/`: core modules (config, data parsing/mapping, models, training, evaluation, utilities).
- `train.py`: main CLI for AG+HC experiments.
- `train_hc.py`: HC-focused CLI (train on HC+AG, test on HC).
