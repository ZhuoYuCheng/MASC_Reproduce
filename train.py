# filename: train.py
# MASC-style anomaly localization for Who&When (paper-aligned, unsupervised)
# - Compatible with Algorithm-Generated (system_prompt + name + is_correct) and Hand-Crafted (role-only + is_corrected)
# - Trains only on normal trajectories (or pseudo-normal trajectories generated via LLM prompting)
# - Optimizes L_recon + lambda * L_proto; anomaly score uses alpha/beta weighting
# - Keeps prototype-guided enhancement to stabilize under sparse context

import argparse
import os
import re
import glob
import json
import random
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 0) Config
# ============================================================
@dataclass
class Config:
    sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_id: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    cache_dir: str = "./.model_cache"
    local_files_only: bool = True
    llm_stub: bool = False

    data_base_path: str = "./Who&When"
    output_dir: str = "./outputs"

    # Use both subsets for training/eval unless you want to restrict
    train_subsets: List[str] = field(default_factory=lambda: ["Hand-Crafted", "Algorithm-Generated"])
    test_subsets: List[str] = field(default_factory=lambda: ["Algorithm-Generated"])

    seed: int = 42

    epochs: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 16

    embedding_dim: int = 384
    hidden_dim: int = 384

    # regularization weights (make proto small to avoid collapse)
    lambda_proto: float = 0.05

    # score components weights (fed into learnable head, so absolute values less sensitive)
    alpha_l2: float = 5.0
    beta_cos: float = 2.0

    # only train/eval on steps up to mistake (inclusive) to avoid cascaded contamination
    truncate_after_mistake: bool = True

    # skip very-early mistakes: if mistake_idx < this, skip for weak supervision (too little normal context)
    min_mistake_idx_for_supervision: int = 2

    # Hand-Crafted parsing noise control
    skip_orchestrator_thought: bool = True
    skip_orchestrator_route: bool = True
    keep_orchestrator_route_as_agent: bool = True
    max_text_len: int = 1200

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bf16"  # bf16|fp16|fp32

    # pseudo-normal generation (for unsupervised training when no is_correct=True)
    use_pseudo_normal_training: bool = True
    pseudo_normal_max_trajs: int = 600
    pseudo_normal_max_steps: int = 10
    pseudo_normal_max_history_chars: int = 2000
    pseudo_normal_temperature: float = 0.2
    pseudo_normal_top_p: float = 0.7
    pseudo_normal_max_new_tokens: int = 128
    pseudo_normal_cache_path: str = "./.pseudo_normal_cache.jsonl"
    pseudo_normal_use_cache: bool = True
    pseudo_normal_progress_every: int = 20
    pseudo_normal_min_prefix_len: int = 2

    # weak supervision
    supervised_loss_weight: float = 0.5
    weak_train_backbone: bool = True
    content_mask_token: str = "[MASK]"
    max_train_trajs: int = -1
    max_eval_trajs: int = -1
    score_running_norm: bool = True
    score_running_eps: float = 1e-6

    def __post_init__(self):
        print(f"Using device: {self.device}")
        if self.device == "cpu" and self.precision == "fp16":
            self.precision = "fp32"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================
# 1) Model download helper (ModelScope optional)
# ============================================================
def download_or_use_id(model_id: str, cache_dir: str) -> str:
    try:
        from modelscope import snapshot_download
        return snapshot_download(model_id=model_id, cache_dir=cache_dir)
    except Exception:
        return model_id


def resolve_model_dir(model_id: str, cache_dir: str, local_roots: List[str]) -> str:
    if os.path.isdir(model_id):
        return model_id
    for root in local_roots:
        cand = os.path.join(root, model_id)
        if os.path.isdir(cand):
            return cand
    return download_or_use_id(model_id, cache_dir)


# ============================================================
# 2) Parsing (compatible)
# ============================================================
def _get_is_correct(item: dict) -> bool:
    if "is_correct" in item:
        return bool(item.get("is_correct"))
    if "is_corrected" in item:
        return bool(item.get("is_corrected"))
    return False


def _clean_content(text: str, max_len: int) -> str:
    if not text:
        return ""
    if "Traceback (most recent call last)" in text:
        text = text.split("Traceback (most recent call last)")[0]

    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("http://") or s.startswith("https://"):
            continue
        lines.append(ln)
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > max_len:
        text = text[:max_len] + " ..."
    return text


def _extract_agent_from_role(role: str) -> str:
    role = (role or "").strip()
    if not role:
        return "Unknown"
    if "(->" in role:
        base = role.split("(")[0].strip()
        return base if base else "Orchestrator"
    base = role.split("(")[0].strip()
    if base.lower() in ["human", "user"]:
        return "User"
    if base.lower() == "assistant":
        return "Assistant"
    return base if base else role


def _norm_agent(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _should_keep_file(path: str, subsets: List[str]) -> bool:
    if not subsets:
        return True
    return any(s in path for s in subsets)


def _infer_subset(file_path: str) -> str:
    if "Hand-Crafted" in file_path:
        return "Hand-Crafted"
    if "Algorithm-Generated" in file_path:
        return "Algorithm-Generated"
    return "Unknown"


def parse_one_item_to_trajectory(item: dict, file_path: str, cfg: Config) -> Optional[dict]:
    query = item.get("question")
    if not query:
        for h in item.get("history", []) or []:
            if (h.get("role") or "").lower() in ["human", "user"]:
                query = h.get("content")
                break
    if not query:
        return None

    is_correct = _get_is_correct(item)
    history = item.get("history", []) or []

    system_prompts = item.get("system_prompt", {}) or {}
    known_agents = set(system_prompts.keys())  # non-empty for algorithm-generated
    agent_role_prompts: Dict[str, str] = dict(system_prompts)

    agent_steps: List[Tuple[str, str]] = []
    raw_to_agent_step: Dict[int, int] = {}

    for raw_idx, h in enumerate(history):
        content = _clean_content(h.get("content", "") or "", cfg.max_text_len)
        if not content:
            continue

        if h.get("name"):
            agent_name = h.get("name")
        else:
            agent_name = _extract_agent_from_role(h.get("role", ""))

        # Algorithm-Generated: treat only system_prompt keys as agents; others as OBS/tool output
        if known_agents:
            if agent_name in known_agents:
                if agent_name not in agent_role_prompts:
                    agent_role_prompts[agent_name] = system_prompts.get(agent_name, "")
                agent_steps.append((agent_name, content))
                raw_to_agent_step[raw_idx] = len(agent_steps) - 1
            else:
                if agent_steps:
                    prev_name, prev_content = agent_steps[-1]
                    agent_steps[-1] = (prev_name, prev_content + "\n\n[OBS]\n" + content)
            continue

        # Hand-Crafted: role-only
        role_str = (h.get("role") or "")
        if cfg.skip_orchestrator_thought and role_str.startswith("Orchestrator") and "(thought)" in role_str:
            continue
        if role_str.startswith("Orchestrator") and "(->" in role_str:
            routed = role_str.split("(->", 1)[1].split(")")[0].strip()
            if routed and cfg.keep_orchestrator_route_as_agent:
                agent_name = routed
            elif cfg.skip_orchestrator_route:
                continue
        if agent_name == "User":
            continue

        if agent_name not in agent_role_prompts and role_str:
            agent_role_prompts[agent_name] = role_str
        agent_steps.append((agent_name, content))
        raw_to_agent_step[raw_idx] = len(agent_steps) - 1

    if not agent_steps:
        return None

    mistake_idx = -1
    method_used = "none"

    if not is_correct:
        mistake_agent = item.get("mistake_agent", None)
        mistake_step_raw = item.get("mistake_step", None)

        k = None
        if mistake_step_raw is not None:
            try:
                k = int(mistake_step_raw)
            except Exception:
                k = None
        agent_norms = [_norm_agent(a) for a, _ in agent_steps]
        mistake_agent_norm = _norm_agent(mistake_agent)
        if mistake_agent_norm and mistake_agent_norm not in agent_norms:
            # Common Hand-Crafted case: Orchestrator or case-mismatch; fallback to any-agent mapping.
            mistake_agent_norm = ""

        def accept(idx: int) -> bool:
            if idx < 0 or idx >= len(agent_steps):
                return False
            if mistake_agent_norm:
                return _norm_agent(agent_steps[idx][0]) == mistake_agent_norm
            return True

        if k is not None:
            # A) agent-local
            if mistake_agent_norm:
                pos = [i for i, (nm, _) in enumerate(agent_steps) if _norm_agent(nm) == mistake_agent_norm]
                if 0 <= k < len(pos) and accept(pos[k]):
                    mistake_idx = pos[k]
                    method_used = "A_agent_local"

            # B) agent-global
            if mistake_idx == -1 and 0 <= k < len(agent_steps) and accept(k):
                mistake_idx = k
                method_used = "B_agent_global"

            # C) raw-to-agent
            if mistake_idx == -1:
                candidates = [ri for ri in raw_to_agent_step.keys() if ri <= k]
                if candidates:
                    closest = max(candidates)
                    idx = raw_to_agent_step[closest]
                    if accept(idx):
                        mistake_idx = idx
                        method_used = "C_raw_to_agent"
            # D) raw-to-agent relaxed (agent mismatch or missing)
            if mistake_idx == -1:
                candidates = [ri for ri in raw_to_agent_step.keys() if ri <= k]
                if candidates:
                    closest = max(candidates)
                    mistake_idx = raw_to_agent_step[closest]
                    method_used = "D_raw_to_agent_relaxed"
            # E) raw-to-agent closest (allow mapping to future step if none before)
            if mistake_idx == -1 and raw_to_agent_step:
                closest = min(raw_to_agent_step.keys(), key=lambda ri: abs(ri - k))
                mistake_idx = raw_to_agent_step[closest]
                method_used = "E_raw_to_agent_closest"

    return {
        "query": _clean_content(query, cfg.max_text_len),
        "steps": agent_steps,
        "is_correct": is_correct,
        "mistake_idx": mistake_idx,
        "mistake_agent": item.get("mistake_agent", None),
        "mistake_step_raw": item.get("mistake_step", None),
        "file_path": file_path,
        "map_method": method_used,
        "agent_roles": agent_role_prompts,
        "subset": _infer_subset(file_path),
    }


def load_trajectories(base: str, subsets: List[str], cfg: Config) -> List[dict]:
    files = glob.glob(os.path.join(base, "**", "*.json"), recursive=True)
    files = [p for p in files if _should_keep_file(p, subsets)]
    print(f"Found {len(files)} files for subsets={subsets}")

    trajs: List[dict] = []
    incorrect = 0
    incorrect_bad = 0
    methods: Dict[str, int] = {}

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = json.load(f)
            items = content if isinstance(content, list) else [content]
            for it in items:
                tr = parse_one_item_to_trajectory(it, fp, cfg)
                if tr is None:
                    continue
                trajs.append(tr)
                methods[tr["map_method"]] = methods.get(tr["map_method"], 0) + 1
                if not tr["is_correct"]:
                    incorrect += 1
                    if tr["mistake_idx"] == -1:
                        incorrect_bad += 1
        except Exception:
            continue

    if incorrect > 0:
        top = sorted(methods.items(), key=lambda x: -x[1])
        print(f"[DataDiag] incorrect={incorrect}, incorrect mistake_idx=-1={incorrect_bad}")
        print("[DataDiag] mapping methods:", ", ".join([f"{k}:{v}" for k, v in top[:8]]))
    return trajs


def print_basic_stats(tag: str, trajs: List[dict], cfg: Config):
    total = len(trajs)
    ok = sum(1 for t in trajs if t["is_correct"])
    bad = total - ok
    known = sum(1 for t in trajs if (not t["is_correct"]) and t["mistake_idx"] != -1)
    unk = sum(1 for t in trajs if (not t["is_correct"]) and t["mistake_idx"] == -1)
    lens = [len(t["steps"]) for t in trajs]
    avg_len = float(np.mean(lens)) if lens else 0.0
    mids = [t["mistake_idx"] for t in trajs if (not t["is_correct"]) and t["mistake_idx"] != -1]
    mid_ge2 = sum(1 for m in mids if m >= cfg.min_mistake_idx_for_supervision)
    print(f"\n[{tag}] total={total} | correct={ok} | incorrect={bad}")
    print(f"[{tag}] incorrect known_mistake={known} | mistake_idx=-1={unk} | eligible(m>= {cfg.min_mistake_idx_for_supervision})={mid_ge2}")
    print(f"[{tag}] avg steps/trajectory={avg_len:.2f}")


def _iter_json_items(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content if isinstance(content, list) else [content]


def run_audit(cfg: Config, sample_failures: int = 5) -> Dict[str, Any]:
    files = glob.glob(os.path.join(cfg.data_base_path, "**", "*.json"), recursive=True)
    subsets = {"Hand-Crafted": [], "Algorithm-Generated": [], "Unknown": []}
    for fp in files:
        subsets[_infer_subset(fp)].append(fp)

    audit: Dict[str, Any] = {
        "summary": {},
        "subsets": {},
        "mapping_methods": {},
        "mistake_idx": {},
        "fail_samples": [],
    }

    all_trajs: List[dict] = []
    for subset, fps in subsets.items():
        raw_items = 0
        true_correct = 0
        true_corrected = 0
        trajs = []
        for fp in fps:
            try:
                items = _iter_json_items(fp)
            except Exception:
                continue
            raw_items += len(items)
            for it in items:
                if it.get("is_correct") is True:
                    true_correct += 1
                if it.get("is_corrected") is True:
                    true_corrected += 1
                tr = parse_one_item_to_trajectory(it, fp, cfg)
                if tr is not None:
                    trajs.append(tr)
        all_trajs.extend(trajs)
        audit["subsets"][subset] = {
            "file_count": len(fps),
            "raw_items": raw_items,
            "parsed_trajectories": len(trajs),
            "is_correct_true": true_correct,
            "is_corrected_true": true_corrected,
        }

    methods: Dict[str, int] = {}
    mistake_idx_values = []
    step_lens = []
    mistake_hist: Dict[str, int] = {}
    incorrect = 0
    incorrect_bad = 0
    failures = 0

    for tr in all_trajs:
        methods[tr["map_method"]] = methods.get(tr["map_method"], 0) + 1
        step_lens.append(len(tr["steps"]))
        if not tr["is_correct"]:
            incorrect += 1
            if tr["mistake_idx"] == -1:
                incorrect_bad += 1
                if failures < sample_failures:
                    failures += 1
                    steps_preview = tr["steps"][:4]
                    audit["fail_samples"].append(
                        {
                            "file_path": tr["file_path"],
                            "subset": tr["subset"],
                            "mistake_agent": tr.get("mistake_agent"),
                            "mistake_step_raw": tr.get("mistake_step_raw"),
                            "steps_preview": [
                                {"agent": a, "content_head": c[:120]}
                                for a, c in steps_preview
                            ],
                        }
                    )
            else:
                mistake_idx_values.append(tr["mistake_idx"])
                k = str(tr["mistake_idx"])
                mistake_hist[k] = mistake_hist.get(k, 0) + 1

    total_trajs = len(all_trajs)
    ok = sum(1 for t in all_trajs if t["is_correct"])
    audit["summary"] = {
        "total_trajectories": total_trajs,
        "correct": ok,
        "incorrect": total_trajs - ok,
        "incorrect_mistake_idx_mapped": incorrect - incorrect_bad,
        "incorrect_mistake_idx_failed": incorrect_bad,
    }
    audit["mapping_methods"] = methods
    if step_lens:
        audit["step_len"] = {
            "min": int(np.min(step_lens)),
            "max": int(np.max(step_lens)),
            "mean": float(np.mean(step_lens)),
            "median": float(np.median(step_lens)),
        }
    if mistake_idx_values:
        audit["mistake_idx"] = {
            "min": int(np.min(mistake_idx_values)),
            "max": int(np.max(mistake_idx_values)),
            "mean": float(np.mean(mistake_idx_values)),
            "median": float(np.median(mistake_idx_values)),
            "hist": mistake_hist,
        }
    return audit


# ============================================================
# 3) Models
# ============================================================
class ContextEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.device = cfg.device
        sentence_model_dir = resolve_model_dir(
            cfg.sentence_model_id,
            cfg.cache_dir,
            local_roots=["./models/sbert", "./.model_cache"],
        )
        self.embedder = SentenceTransformer(
            sentence_model_dir,
            cache_folder=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
        ).to(self.device)
        for p in self.embedder.parameters():
            p.requires_grad_(False)

        self.f_q = nn.Linear(cfg.embedding_dim, cfg.hidden_dim)
        self.f_h = nn.Linear(cfg.embedding_dim * 2, cfg.hidden_dim)

    @torch.no_grad()
    def _encode(self, text_or_texts) -> torch.Tensor:
        emb = self.embedder.encode(text_or_texts, convert_to_tensor=True).to(self.device)
        return emb.detach().clone()

    def forward(self, query: str, hist: List[Tuple[str, str]]):
        q_emb = self._encode(query)
        q_tilde = self.f_q(q_emb)
        h_tilde = None
        if hist:
            names = [a for a, _ in hist]
            conts = [c for _, c in hist]
            r = self._encode(names)
            x = self._encode(conts)
            h_tilde = self.f_h(torch.cat([r, x], dim=-1))
        return q_tilde, h_tilde

    def step_gt(self, agent_name: str, content: str) -> torch.Tensor:
        r = self._encode(agent_name)
        x = self._encode(content)
        return self.f_h(torch.cat([r, x], dim=-1)).unsqueeze(0)


class PrototypeUpdater(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

    def forward(self, p: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        q = self.Wq(p)  # [1,d]
        k = self.Wk(X)  # [T,d]
        v = self.Wv(X)  # [T,d]
        att = torch.softmax((q @ k.T) / (p.size(-1) ** 0.5), dim=-1)  # [1,T]
        return att @ v


class MASC(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.llm_stub = cfg.llm_stub
        dtype = (
            torch.bfloat16 if cfg.precision == "bf16"
            else (torch.float16 if cfg.precision == "fp16" else torch.float32)
        )

        if self.llm_stub:
            print("Using stub backbone (GRU) for CPU-friendly debug runs.")
            llm_dim = cfg.hidden_dim
            self.llm = nn.GRU(llm_dim, llm_dim, batch_first=True)
            self.adapter_in = nn.Identity()
            self.f_theta = nn.Identity()
        else:
            llm_model_dir = resolve_model_dir(
                cfg.llm_model_id,
                cfg.cache_dir,
                local_roots=["./models/llm", "./.model_cache"],
            )
            print(f"Loading LLM Backbone: {llm_model_dir}")
            self.llm = AutoModel.from_pretrained(
                llm_model_dir,
                trust_remote_code=True,
                dtype=dtype,
                local_files_only=cfg.local_files_only,
            ).to(cfg.device)

            for p in self.llm.parameters():
                p.requires_grad_(False)

            llm_conf = self.llm.config
            llm_dim = getattr(llm_conf, "n_embd", getattr(llm_conf, "hidden_size", 4096))

            self.adapter_in = nn.Linear(cfg.hidden_dim, llm_dim)
            self.f_theta = nn.Linear(llm_dim, cfg.hidden_dim)

        self.prototype = nn.Parameter(torch.randn(1, cfg.hidden_dim))
        self.proto_updater = PrototypeUpdater(cfg.hidden_dim)

    def forward(self, q_tilde: torch.Tensor, h_tilde: Optional[torch.Tensor]) -> torch.Tensor:
        if h_tilde is None:
            seq = q_tilde.unsqueeze(0)
        else:
            seq = torch.cat([q_tilde.unsqueeze(0), h_tilde], dim=0)
        llm_in = self.adapter_in(seq).unsqueeze(0)
        if self.llm_stub:
            out, _ = self.llm(llm_in)
            last = out[:, -1, :]
            return self.f_theta(last)
        llm_in = llm_in.to(self.llm.dtype)
        out = self.llm(inputs_embeds=llm_in)
        last = out.last_hidden_state[:, -1, :]
        return self.f_theta(last.to(self.f_theta.weight.dtype))

    def proto_ctx(self, xhat_hist_detached: List[torch.Tensor]) -> torch.Tensor:
        if not xhat_hist_detached:
            return self.prototype
        X = torch.cat(xhat_hist_detached, dim=0)
        return self.proto_updater(self.prototype, X)


class StepScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, l2: torch.Tensor, dcos: torch.Tensor) -> torch.Tensor:
        x = torch.stack([l2, dcos], dim=-1)
        return self.linear(x).squeeze(-1)


# ============================================================
# 4) Pseudo-normal generation (LLM prompted)
# ============================================================
class PseudoNormalGenerator:
    def __init__(self, cfg: Config):
        self.device = cfg.device
        dtype = (
            torch.bfloat16 if cfg.precision == "bf16"
            else (torch.float16 if cfg.precision == "fp16" else torch.float32)
        )
        llm_model_dir = resolve_model_dir(
            cfg.llm_model_id,
            cfg.cache_dir,
            local_roots=["./models/llm", "./.model_cache"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_dir,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_dir,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=cfg.local_files_only,
        ).to(cfg.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _build_prompt(
        self,
        query: str,
        agent_name: str,
        agent_role_prompt: str,
        history_steps: List[Tuple[str, str]],
        cfg: Config,
    ) -> str:
        hist_lines = []
        for name, content in history_steps:
            hist_lines.append(f"{name}: {content}")
        history = "\n\n".join(hist_lines).strip()
        if len(history) > cfg.pseudo_normal_max_history_chars:
            history = history[-cfg.pseudo_normal_max_history_chars :]

        role_block = agent_role_prompt.strip()
        if not role_block:
            role_block = "(none)"

        user_body = (
            f"Task Query:\n{query}\n\n"
            f"Agent Name: {agent_name}\n"
            f"Agent Role Description:\n{role_block}\n\n"
            f"History (previous steps, most recent last):\n{history if history else '(none)'}\n\n"
            "Instruction: Produce the next step output for the specified agent. "
            "Only output the agent response text. Do not add role labels or explanations."
        )

        sys_inst = (
            "You generate a correct and coherent next-step output for a multi-agent system. "
            "Follow the agent role description if provided. Output only the agent response text."
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": user_body},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return sys_inst + "\n\n" + user_body + "\n\nAnswer:\n"

    def generate_step_output(
        self,
        query: str,
        agent_name: str,
        agent_role_prompt: str,
        history_steps: List[Tuple[str, str]],
        cfg: Config,
        cache: Dict[str, str],
    ) -> str:
        prompt = self._build_prompt(query, agent_name, agent_role_prompt, history_steps, cfg)
        key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=cfg.pseudo_normal_temperature,
                top_p=cfg.pseudo_normal_top_p,
                max_new_tokens=cfg.pseudo_normal_max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not text:
            text = "(empty)"
        cache[key] = text
        return text


def _load_pseudo_cache(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    cache: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "key" in rec and "output" in rec:
                    cache[rec["key"]] = rec["output"]
            except Exception:
                continue
    return cache


def _append_pseudo_cache(path: str, key: str, output: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "output": output}, ensure_ascii=True) + "\n")


def build_pseudo_normal_trajectories(trajs: List[dict], cfg: Config) -> List[dict]:
    print("[PseudoNormal] Building pseudo-normal trajectories via LLM prompting...")
    gen = PseudoNormalGenerator(cfg)
    cache = _load_pseudo_cache(cfg.pseudo_normal_cache_path) if cfg.pseudo_normal_use_cache else {}
    if cfg.pseudo_normal_use_cache:
        print(f"[PseudoNormal] Cache loaded: {len(cache)} entries from {cfg.pseudo_normal_cache_path}")
    built: List[dict] = []
    total = 0
    total_seen = 0

    for tr in trajs:
        total_seen += 1
        if total >= cfg.pseudo_normal_max_trajs:
            break
        q = tr["query"]
        steps = tr["steps"]
        agent_roles = tr.get("agent_roles", {}) or {}

        new_steps: List[Tuple[str, str]] = []
        hist_so_far: List[Tuple[str, str]] = []
        max_steps = min(cfg.pseudo_normal_max_steps, len(steps))
        for t in range(max_steps):
            agent_name, _ = steps[t]
            role_prompt = agent_roles.get(agent_name, "")
            prompt = gen._build_prompt(q, agent_name, role_prompt, hist_so_far, cfg)
            key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
            if key in cache:
                out = cache[key]
            else:
                out = gen.generate_step_output(q, agent_name, role_prompt, hist_so_far, cfg, cache)
                _append_pseudo_cache(cfg.pseudo_normal_cache_path, key, out)
            new_steps.append((agent_name, _clean_content(out, cfg.max_text_len)))
            hist_so_far.append((agent_name, new_steps[-1][1]))

        if not new_steps:
            continue

        built.append(
            {
                "query": q,
                "steps": new_steps,
                "is_correct": True,
                "mistake_idx": -1,
                "file_path": tr["file_path"],
                "map_method": "pseudo_normal",
                "agent_roles": agent_roles,
                "pseudo_normal": True,
            }
        )
        total += 1
        if cfg.pseudo_normal_progress_every > 0 and (total % cfg.pseudo_normal_progress_every == 0):
            print(f"[PseudoNormal] progress: built {total} / max {cfg.pseudo_normal_max_trajs} (seen {total_seen})")

    print(f"[PseudoNormal] Built {len(built)} trajectories (max {cfg.pseudo_normal_max_trajs}).")
    if cfg.pseudo_normal_use_cache:
        print(f"[PseudoNormal] Cache size after build: {len(cache)} entries")
    return built


def build_prefix_normal_trajectories(trajs: List[dict], cfg: Config) -> List[dict]:
    """Use pre-mistake prefixes as pseudo-normal samples without using labels as supervision."""
    built: List[dict] = []
    for tr in trajs:
        if tr["is_correct"]:
            continue
        m = tr.get("mistake_idx", -1)
        if m is None or m < cfg.pseudo_normal_min_prefix_len:
            continue
        prefix_steps = tr["steps"][:m]
        if not prefix_steps:
            continue
        built.append(
            {
                "query": tr["query"],
                "steps": prefix_steps,
                "is_correct": True,
                "mistake_idx": -1,
                "file_path": tr["file_path"],
                "map_method": "prefix_normal",
                "agent_roles": tr.get("agent_roles", {}),
                "pseudo_normal": True,
            }
        )
    return built


# ============================================================
# 5) Training
# ============================================================
def init_prototype_simple(encoder: ContextEncoder, masc: MASC, trajs: List[dict], max_steps: int = 800):
    """Initialize prototype as mean of early steps (best-effort)."""
    encoder.eval()
    masc.eval()
    embs = []
    with torch.no_grad():
        for tr in trajs:
            steps = tr["steps"]
            for agent_name, content in steps[:3]:  # only early context
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


def train(trajs: List[dict], encoder: ContextEncoder, masc: MASC, cfg: Config):
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


# ============================================================
# 6) Evaluation
# ============================================================
def _apply_ablation(trajs: List[dict], ablation: Optional[str], cfg: Config, seed: int) -> List[dict]:
    if not ablation:
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


def evaluate_auc(
    trajs: List[dict],
    encoder: ContextEncoder,
    masc: MASC,
    cfg: Config,
    scorer: Optional[StepScorer] = None,
    truncate_after_mistake: Optional[bool] = None,
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

    eval_trajs = _apply_ablation(trajs, ablation, cfg, seed)
    trunc = cfg.truncate_after_mistake if truncate_after_mistake is None else truncate_after_mistake

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
                else:
                    if cfg.score_running_norm and len(l2_hist) >= 2:
                        l2_mean = float(np.mean(l2_hist))
                        l2_std = float(np.std(l2_hist) + cfg.score_running_eps)
                        dcos_mean = float(np.mean(dcos_hist))
                        dcos_std = float(np.std(dcos_hist) + cfg.score_running_eps)
                        z_l2 = (float(l2.item()) - l2_mean) / l2_std
                        z_cos = (float(dcos.item()) - dcos_mean) / dcos_std
                        logit = float(z_l2 + z_cos)
                    else:
                        logit = float((cfg.alpha_l2 * l2 + cfg.beta_cos * dcos).item())

                y = 1 if t == m else 0
                labels.append(y)
                logits.append(logit)

                hist_so_far.append((agent_name, content))
                xhat_hist_detached.append(x_hat.detach())
                l2_hist.append(float(l2.item()))
                dcos_hist.append(float(dcos.item()))

            used_trajs += 1

    labels_arr = np.array(labels, dtype=np.int64)
    logits_arr = np.array(logits, dtype=np.float64)
    stats = {
        "total_steps": int(len(labels_arr)),
        "anomaly_steps": int(labels_arr.sum()),
        "normal_steps": int(len(labels_arr) - labels_arr.sum()),
        "used_trajs": int(used_trajs),
        "truncate_after_mistake": bool(trunc),
        "ablation": ablation or "none",
    }

    if labels_arr.sum() == 0 or labels_arr.sum() == len(labels_arr):
        stats["auc_pos"] = None
        stats["auc_neg"] = None
        stats["selected_auc"] = None
        stats["score_direction"] = None
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
    return stats


# ============================================================
# 7) Main
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json(obj: Dict[str, Any], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)
    print(f"[Output] Saved {out_path}")


def _prepare_unsup_training(trajs: List[dict], cfg: Config) -> Tuple[List[dict], List[str]]:
    notes = []
    normal_trajs = [t for t in trajs if t["is_correct"]]
    if normal_trajs:
        notes.append(f"normal_trajectories={len(normal_trajs)}")
        return normal_trajs, notes

    # Paper-aligned unsupervised assumes normal trajectories; we fallback to prefix mining when none exist.
    prefix_trajs = build_prefix_normal_trajectories(trajs, cfg)
    if prefix_trajs:
        notes.append("no_true_normal: using pre-mistake prefixes as pseudo-normal")
        notes.append(f"prefix_trajs={len(prefix_trajs)}")
        return prefix_trajs, notes

    if cfg.use_pseudo_normal_training:
        # Deviation: LLM-generated pseudo-normal trajectories when no clean normals exist.
        if cfg.llm_stub:
            raise RuntimeError("llm_stub does not support pseudo-normal generation.")
        notes.append("no_true_normal: using LLM-generated pseudo-normal")
        pseudo = build_pseudo_normal_trajectories(trajs, cfg)
        return pseudo, notes

    raise RuntimeError("No is_correct=True trajectories and pseudo-normal training disabled.")


def main():
    parser = argparse.ArgumentParser(description="MASC reproduction: audit/unsupervised/weak supervision")
    parser.add_argument("--mode", choices=["audit", "unsup", "weak"], default="unsup")
    parser.add_argument("--ablate", choices=["none", "agent_only", "content_mask", "shuffle_steps", "cross_subset", "all"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_subsets", type=str, default="Hand-Crafted,Algorithm-Generated")
    parser.add_argument("--test_subsets", type=str, default="Algorithm-Generated")
    parser.add_argument("--max_train_trajs", type=int, default=-1)
    parser.add_argument("--max_eval_trajs", type=int, default=-1)
    parser.add_argument("--truncate_after_mistake", action="store_true")
    parser.add_argument("--no_truncate_after_mistake", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--llm_stub", action="store_true")
    parser.add_argument("--no_score_running_norm", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.train_subsets = [s.strip() for s in args.train_subsets.split(",") if s.strip()]
    cfg.test_subsets = [s.strip() for s in args.test_subsets.split(",") if s.strip()]
    cfg.max_train_trajs = args.max_train_trajs
    cfg.max_eval_trajs = args.max_eval_trajs
    if args.truncate_after_mistake:
        cfg.truncate_after_mistake = True
    if args.no_truncate_after_mistake:
        cfg.truncate_after_mistake = False
    if args.allow_download:
        cfg.local_files_only = False
    if args.local_files_only:
        cfg.local_files_only = True
    if args.llm_stub:
        cfg.llm_stub = True
    if args.no_score_running_norm:
        cfg.score_running_norm = False

    set_seed(cfg.seed)

    if args.mode == "audit":
        audit = run_audit(cfg)
        out_path = os.path.join(cfg.output_dir, f"audit_{int(time.time())}.json")
        _save_json(audit, out_path)
        print("[Audit] Summary:", audit.get("summary"))
        return

    print("Initializing Models...")
    encoder = ContextEncoder(cfg).to(cfg.device)
    masc = MASC(cfg).to(cfg.device)
    scorer = StepScorer().to(cfg.device)

    train_trajs = load_trajectories(cfg.data_base_path, cfg.train_subsets, cfg)
    test_trajs = load_trajectories(cfg.data_base_path, cfg.test_subsets, cfg)
    if cfg.max_train_trajs > 0:
        train_trajs = train_trajs[: cfg.max_train_trajs]
        print(f"[Train] Truncated train_trajs to {len(train_trajs)}")
    if cfg.max_eval_trajs > 0:
        test_trajs = test_trajs[: cfg.max_eval_trajs]
        print(f"[Eval] Truncated test_trajs to {len(test_trajs)}")

    print_basic_stats("TRAIN", train_trajs, cfg)
    print_basic_stats("TEST", test_trajs, cfg)

    if args.mode == "unsup":
        train_trajs, notes = _prepare_unsup_training(train_trajs, cfg)
        print("[Train] Unsupervised policy:", "; ".join(notes))

        init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        train(train_trajs, encoder, masc, cfg)

        out = {"mode": "unsup", "notes": notes, "seed": cfg.seed}
        for trunc in [True, False]:
            stats = evaluate_auc(test_trajs, encoder, masc, cfg, scorer=None, truncate_after_mistake=trunc)
            print(f"\n--- Evaluation (unsup, truncate={trunc}) ---")
            print(stats)
            out[f"eval_truncate_{trunc}"] = stats
        out_path = os.path.join(cfg.output_dir, f"unsup_{int(time.time())}.json")
        _save_json(out, out_path)
        return

    if args.mode == "weak":
        init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        train_weak(train_trajs, encoder, masc, scorer, cfg)

        out = {"mode": "weak", "seed": cfg.seed}
        for trunc in [True, False]:
            stats = evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer, truncate_after_mistake=trunc)
            print(f"\n--- Evaluation (weak, truncate={trunc}) ---")
            print(stats)
            out[f"eval_truncate_{trunc}"] = stats

        if args.ablate != "none":
            ablate_results = {}
            if args.ablate == "all":
                ablations = ["agent_only", "content_mask", "shuffle_steps", "cross_subset"]
            else:
                ablations = [args.ablate]

            for ab in ablations:
                if ab == "cross_subset":
                    cross = {}
                    for train_subset, test_subset in [
                        ("Hand-Crafted", "Algorithm-Generated"),
                        ("Algorithm-Generated", "Hand-Crafted"),
                    ]:
                        cfg.train_subsets = [train_subset]
                        cfg.test_subsets = [test_subset]
                        train_trajs = load_trajectories(cfg.data_base_path, cfg.train_subsets, cfg)
                        test_trajs = load_trajectories(cfg.data_base_path, cfg.test_subsets, cfg)
                        if cfg.max_train_trajs > 0:
                            train_trajs = train_trajs[: cfg.max_train_trajs]
                        if cfg.max_eval_trajs > 0:
                            test_trajs = test_trajs[: cfg.max_eval_trajs]
                        init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
                        train_weak(train_trajs, encoder, masc, scorer, cfg)
                        cross_key = f"{train_subset}_to_{test_subset}"
                        cross[cross_key] = evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer)
                    ablate_results[ab] = cross
                else:
                    ablate_results[ab] = evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer, ablation=ab)

            out["ablation"] = ablate_results
            ab_path = os.path.join(cfg.output_dir, f"ablation_{int(time.time())}.json")
            _save_json(out, ab_path)
        else:
            out_path = os.path.join(cfg.output_dir, f"weak_{int(time.time())}.json")
            _save_json(out, out_path)

        return


if __name__ == "__main__":
    main()
