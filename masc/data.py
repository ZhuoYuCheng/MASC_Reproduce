import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .models import PseudoNormalGenerator


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
    if cfg.use_ground_truth:
        gt = item.get("ground_truth")
        if gt:
            gt_clean = _clean_content(gt, cfg.max_text_len)
            query = f"{query}\n\n[GROUND_TRUTH]\n{gt_clean}"

    is_correct = _get_is_correct(item)
    history = item.get("history", []) or []

    system_prompts = item.get("system_prompt", {}) or {}
    known_agents = set(system_prompts.keys())
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
            mistake_agent_norm = ""

        def accept(idx: int) -> bool:
            if idx < 0 or idx >= len(agent_steps):
                return False
            if mistake_agent_norm:
                return _norm_agent(agent_steps[idx][0]) == mistake_agent_norm
            return True

        if k is not None:
            if mistake_agent_norm:
                pos = [i for i, (nm, _) in enumerate(agent_steps) if _norm_agent(nm) == mistake_agent_norm]
                if 0 <= k < len(pos) and accept(pos[k]):
                    mistake_idx = pos[k]
                    method_used = "A_agent_local"

            if mistake_idx == -1 and 0 <= k < len(agent_steps) and accept(k):
                mistake_idx = k
                method_used = "B_agent_global"

            if mistake_idx == -1:
                candidates = [ri for ri in raw_to_agent_step.keys() if ri <= k]
                if candidates:
                    closest = max(candidates)
                    idx = raw_to_agent_step[closest]
                    if accept(idx):
                        mistake_idx = idx
                        method_used = "C_raw_to_agent"

            if mistake_idx == -1:
                candidates = [ri for ri in raw_to_agent_step.keys() if ri <= k]
                if candidates:
                    closest = max(candidates)
                    mistake_idx = raw_to_agent_step[closest]
                    method_used = "D_raw_to_agent_relaxed"

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


def build_prefix_normal_trajectories(trajs: List[dict], cfg: Config) -> List[dict]:
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
