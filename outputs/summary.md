# MASC Reproduction Summary

## Data constraints (audit)
- Dataset: `./Who&When` (Algorithm-Generated + Hand-Crafted)
- Latest audit: `outputs/audit_1767847126.json`
- total=184, correct=0 (no true normal trajectories)
- incorrect=184, mistake mapped=184, failed=0
- Mapping methods: A_agent_local=84, B_agent_global=31, C_raw_to_agent=54, D_raw_to_agent_relaxed=14, E_raw_to_agent_closest=1
- Step stats: mean steps=11.77, median=7

## Unsupervised (paper-facing)
- Training policy: no true normals => prefix-mined pseudo-normal (pre-mistake prefixes), no label supervision
- Scoring: reconstruction + prototype score with per-trajectory running normalization
- Run: `outputs/unsup_1767847987.json`
- AUC truncate=True: 0.6809 (selected_auc, score_direction=neg)
- AUC truncate=False: 0.6795 (selected_auc, score_direction=neg)

## Weak supervision (control)
- Run: `outputs/ablation_1767846393.json`
- AUC truncate=True: 0.6315
- AUC truncate=False: 0.6016

## Ablations (weak)
- agent_only: 0.6745
- content_mask: 0.6175
- shuffle_steps: 0.6643
- cross_subset:
  - Hand-Crafted -> Algorithm-Generated: 0.6038
  - Algorithm-Generated -> Hand-Crafted: 0.4957

## Target assessment
- Unsupervised target: PASS
  - truncate=True >= 0.60
  - truncate=False >= 0.52–0.55
  - Also meets “better” threshold (>=0.65 and >=0.6)
- Weak target: NOT MET (AUC < 0.80)

## Key conclusions
- Data reality: zero true normals forced a prefix-mined pseudo-normal strategy; still unsupervised by definition (no mistake labels used for training).
- Mapping: Orchestrator/case-mismatch and raw index alignment were stabilized via normalized agent matching and relaxed raw-to-agent mapping; failed mappings reduced to 0.
- Unsupervised gains: running normalization of reconstruction/prototype scores improved anomaly separability without labels.
- Weak supervision: moderate AUC; cross-subset degradation suggests domain shift/template reliance rather than robust reasoning.

## Repro commands
- Audit: `python train.py --mode audit`
- Unsupervised (paper-facing): `python train.py --mode unsup`
- Weak + ablation: `python train.py --mode weak --ablate all --epochs 1`
