import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .data import build_prefix_normal_trajectories, build_pseudo_normal_trajectories
from .models import ContextEncoder, MASC, StepScorer


def init_prototype_simple(encoder: ContextEncoder, masc: MASC, trajs: List[dict], max_steps: int = 800):
    encoder.eval()
    masc.eval()
    embs = []
    with torch.no_grad():
        for tr in trajs:
            steps = tr["steps"]
            for agent_name, content in steps[:3]:
                gt = encoder.step_gt(agent_name, content).squeeze(0)
                embs.append(gt)
                if len(embs) >= max_steps:
                    break
            if len(embs) >= max_steps:
                break
    if embs:
        mean_emb = torch.stack(embs, dim=0).mean(dim=0, keepdim=True)
        masc.prototype.data.copy_(mean_emb.to(masc.prototype.dtype))
        print(f"[Init] Prototype initialized from {len(embs)} early steps.")
    else:
        print("[Init] WARNING: no steps for prototype init; keep random.")


def train_unsup(trajs: List[dict], encoder: ContextEncoder, masc: MASC, cfg: Config):
    encoder.train()
    masc.train()

    params = [p for p in list(encoder.parameters()) + list(masc.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("\n--- Training (unsupervised, normal trajectories only) ---")
    for epoch in range(cfg.epochs):
        random.shuffle(trajs)
        total_loss = 0.0
        total_steps = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        for tr in trajs:
            q = tr["query"]
            steps = tr["steps"]

            hist_so_far: List[Tuple[str, str]] = []
            xhat_hist_detached: List[torch.Tensor] = []
            for agent_name, content in steps:

                q_tilde, h_tilde = encoder(q, hist_so_far)
                gt = encoder.step_gt(agent_name, content)
                x_hat = masc(q_tilde, h_tilde)

                p_ctx = masc.proto_ctx(xhat_hist_detached)
                loss_recon = F.mse_loss(x_hat, gt)
                loss_proto = (1 - F.cosine_similarity(x_hat, p_ctx).mean())
                loss = loss_recon + cfg.lambda_proto * loss_proto

                (loss / cfg.grad_accum_steps).backward()
                accum += 1

                total_loss += float(loss.item())
                total_steps += 1

                hist_so_far.append((agent_name, content))
                xhat_hist_detached.append(x_hat.detach())

                if accum >= cfg.grad_accum_steps:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    accum = 0

        if accum > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        avg = total_loss / max(1, total_steps)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg:.6f} | Steps: {total_steps}")


def train_weak(trajs: List[dict], encoder: ContextEncoder, masc: MASC, scorer: StepScorer, cfg: Config):
    encoder.train(cfg.weak_train_backbone)
    masc.train(cfg.weak_train_backbone)
    scorer.train()

    params = [p for p in scorer.parameters() if p.requires_grad]
    if cfg.weak_train_backbone:
        params += [p for p in list(encoder.parameters()) + list(masc.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("\n--- Training (weak-supervised: recon + BCE) ---")
    for epoch in range(cfg.epochs):
        random.shuffle(trajs)
        total_loss = 0.0
        total_steps = 0
        total_sup = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        for tr in trajs:
            q = tr["query"]
            steps = tr["steps"]
            m = tr.get("mistake_idx", -1)
            use_supervision = (not tr["is_correct"]) and (m is not None) and (m >= 0)

            upto = len(steps)
            if (not tr["is_correct"]) and cfg.truncate_after_mistake and (m is not None) and (m >= 0):
                upto = min(upto, m + 1)

            hist_so_far: List[Tuple[str, str]] = []
            xhat_hist_detached: List[torch.Tensor] = []

            for t in range(upto):
                agent_name, content = steps[t]

                q_tilde, h_tilde = encoder(q, hist_so_far)
                gt = encoder.step_gt(agent_name, content)
                x_hat = masc(q_tilde, h_tilde)

                p_ctx = masc.proto_ctx(xhat_hist_detached)
                loss_recon = F.mse_loss(x_hat, gt)
                loss_proto = (1 - F.cosine_similarity(x_hat, p_ctx).mean())
                loss = loss_recon + cfg.lambda_proto * loss_proto

                if tr["is_correct"]:
                    label = torch.tensor(0.0, device=loss.device)
                elif use_supervision:
                    label = torch.tensor(1.0 if t == m else 0.0, device=loss.device)
                else:
                    label = None

                if label is not None:
                    l2 = F.mse_loss(x_hat, gt)
                    dcos = 1 - F.cosine_similarity(x_hat, p_ctx).mean()
                    logit = scorer(l2, dcos)
                    loss_sup = F.binary_cross_entropy_with_logits(logit, label)
                    loss = loss + cfg.supervised_loss_weight * loss_sup
                    total_sup += 1

                (loss / cfg.grad_accum_steps).backward()
                accum += 1

                total_loss += float(loss.item())
                total_steps += 1

                hist_so_far.append((agent_name, content))
                xhat_hist_detached.append(x_hat.detach())

                if accum >= cfg.grad_accum_steps:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    accum = 0

        if accum > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        avg = total_loss / max(1, total_steps)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg:.6f} | Steps: {total_steps} | SupSteps: {total_sup}")


def prepare_unsup_training(trajs: List[dict], cfg: Config):
    notes = []
    normal_trajs = [t for t in trajs if t["is_correct"]]
    if normal_trajs:
        notes.append(f"normal_trajectories={len(normal_trajs)}")
        return normal_trajs, notes

    prefix_trajs = build_prefix_normal_trajectories(trajs, cfg)
    if prefix_trajs:
        notes.append("no_true_normal: using pre-mistake prefixes as pseudo-normal")
        notes.append(f"prefix_trajs={len(prefix_trajs)}")
        return prefix_trajs, notes

    if cfg.use_pseudo_normal_training:
        if cfg.llm_stub:
            raise RuntimeError("llm_stub does not support pseudo-normal generation.")
        notes.append("no_true_normal: using LLM-generated pseudo-normal")
        pseudo = build_pseudo_normal_trajectories(trajs, cfg)
        return pseudo, notes

    raise RuntimeError("No is_correct=True trajectories and pseudo-normal training disabled.")
