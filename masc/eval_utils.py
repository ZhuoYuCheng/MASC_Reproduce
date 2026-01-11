import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .models import ContextEncoder, MASC, StepScorer


def _apply_ablation(trajs: List[dict], ablation: Optional[str], cfg: Config, seed: int) -> List[dict]:
    if not ablation or ablation == "none":
        return trajs
    rng = random.Random(seed + 7)
    out = []
    for tr in trajs:
        steps = list(tr["steps"])
        m = tr.get("mistake_idx", -1)
        if ablation == "content_mask":
            steps = [(a, cfg.content_mask_token) for a, _ in steps]
        elif ablation == "shuffle_steps":
            step_items = []
            for idx, (a, c) in enumerate(steps):
                step_items.append({"agent": a, "content": c, "is_mistake": idx == m})
            rng.shuffle(step_items)
            steps = [(s["agent"], s["content"]) for s in step_items]
            m = next((i for i, s in enumerate(step_items) if s["is_mistake"]), -1)
        elif ablation == "agent_only":
            target = tr.get("mistake_agent")
            if not target:
                continue
            filtered = [(a, c) for a, c in steps if a == target]
            if not filtered:
                continue
            if m is not None and m >= 0:
                if m >= len(steps) or steps[m][0] != target:
                    continue
                m = sum(1 for a, _ in steps[:m] if a == target)
            steps = filtered

        new_tr = dict(tr)
        new_tr["steps"] = steps
        new_tr["mistake_idx"] = m
        out.append(new_tr)
    return out


def evaluate_metrics(
    trajs: List[dict],
    encoder: ContextEncoder,
    masc: MASC,
    cfg: Config,
    scorer: Optional[StepScorer] = None,
    ablation: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    encoder.eval()
    masc.eval()
    if scorer is not None:
        scorer.eval()

    labels = []
    logits = []
    used_trajs = 0
    traj_scores = []

    eval_trajs = _apply_ablation(trajs, ablation, cfg, seed)
    trunc = True

    with torch.no_grad():
        for tr in eval_trajs:
            if tr["is_correct"]:
                continue
            m = tr["mistake_idx"]
            if m is None or m == -1:
                continue
            if m < cfg.min_mistake_idx_for_supervision:
                continue

            q = tr["query"]
            steps = tr["steps"]
            upto = (m + 1) if trunc else len(steps)

            hist_so_far: List[Tuple[str, str]] = []
            xhat_hist_detached: List[torch.Tensor] = []
            l2_hist: List[float] = []
            dcos_hist: List[float] = []
            l2_all: List[float] = []
            dcos_all: List[float] = []
            step_scores: List[float] = []

            for t in range(min(upto, len(steps))):
                agent_name, content = steps[t]
                q_tilde, h_tilde = encoder(q, hist_so_far)
                gt = encoder.step_gt(agent_name, content)
                x_hat = masc(q_tilde, h_tilde)

                p_ctx = masc.proto_ctx(xhat_hist_detached)
                l2 = F.mse_loss(x_hat, gt)
                dcos = 1 - F.cosine_similarity(x_hat, p_ctx).mean()

                if scorer is not None:
                    logit = float(scorer(l2, dcos).item())
                    step_scores.append(logit)
                else:
                    if cfg.norm_mode == "full":
                        l2_all.append(float(l2.item()))
                        dcos_all.append(float(dcos.item()))
                    elif cfg.norm_mode == "prefix_only" and len(l2_hist) >= cfg.norm_min_steps:
                        l2_mean = float(np.mean(l2_hist))
                        l2_std = float(np.std(l2_hist) + cfg.score_running_eps)
                        dcos_mean = float(np.mean(dcos_hist))
                        dcos_std = float(np.std(dcos_hist) + cfg.score_running_eps)
                        z_l2 = (float(l2.item()) - l2_mean) / l2_std
                        z_cos = (float(dcos.item()) - dcos_mean) / dcos_std
                        step_scores.append(float(z_l2 + z_cos))
                    else:
                        step_scores.append(float((cfg.alpha_l2 * l2 + cfg.beta_cos * dcos).item()))

                hist_so_far.append((agent_name, content))
                xhat_hist_detached.append(x_hat.detach())
                l2_hist.append(float(l2.item()))
                dcos_hist.append(float(dcos.item()))

            if scorer is None and cfg.norm_mode == "full":
                if len(l2_all) >= 2:
                    l2_mean = float(np.mean(l2_all))
                    l2_std = float(np.std(l2_all) + cfg.score_running_eps)
                    dcos_mean = float(np.mean(dcos_all))
                    dcos_std = float(np.std(dcos_all) + cfg.score_running_eps)
                    for l2_v, dcos_v in zip(l2_all, dcos_all):
                        z_l2 = (l2_v - l2_mean) / l2_std
                        z_cos = (dcos_v - dcos_mean) / dcos_std
                        step_scores.append(float(z_l2 + z_cos))
                else:
                    for l2_v, dcos_v in zip(l2_all, dcos_all):
                        step_scores.append(float(cfg.alpha_l2 * l2_v + cfg.beta_cos * dcos_v))

            for t, score in enumerate(step_scores):
                y = 1 if t == m else 0
                labels.append(y)
                logits.append(score)

            traj_scores.append({"mistake_idx": int(m), "scores": step_scores})
            used_trajs += 1

    labels_arr = np.array(labels, dtype=np.int64)
    logits_arr = np.array(logits, dtype=np.float64)
    stats = {
        "total_steps": int(len(labels_arr)),
        "anomaly_steps": int(labels_arr.sum()),
        "normal_steps": int(len(labels_arr) - labels_arr.sum()),
        "used_trajs": int(used_trajs),
        "truncate_after_mistake": True,
        "ablation": ablation or "none",
        "norm_mode": cfg.norm_mode if scorer is None else "scorer",
    }

    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        stats["auc_pos"] = None
        stats["auc_neg"] = None
        stats["selected_auc"] = None
        stats["score_direction"] = None
        stats["accuracy"] = None
        return stats

    from sklearn.metrics import roc_auc_score
    auc_pos = float(roc_auc_score(labels_arr, logits_arr))
    auc_neg = float(roc_auc_score(labels_arr, -logits_arr))
    stats["auc_pos"] = auc_pos
    stats["auc_neg"] = auc_neg
    stats["anom_mean"] = float(logits_arr[labels_arr == 1].mean())
    stats["norm_mean"] = float(logits_arr[labels_arr == 0].mean())
    if scorer is None and stats["anom_mean"] < stats["norm_mean"]:
        stats["selected_auc"] = auc_neg
        stats["score_direction"] = "neg"
    else:
        stats["selected_auc"] = auc_pos
        stats["score_direction"] = "pos"
    correct = 0
    total = 0
    use_min = stats["score_direction"] == "neg"
    for tr in traj_scores:
        if not tr["scores"]:
            continue
        pred_idx = int(np.argmin(tr["scores"]) if use_min else np.argmax(tr["scores"]))
        if pred_idx == tr["mistake_idx"]:
            correct += 1
        total += 1
    stats["accuracy"] = float(correct / total) if total > 0 else None
    stats["accuracy_correct"] = int(correct)
    stats["accuracy_total"] = int(total)
    return stats
