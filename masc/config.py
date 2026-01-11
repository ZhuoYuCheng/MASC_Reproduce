import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_id: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    cache_dir: str = "./.model_cache"
    local_files_only: bool = True
    llm_stub: bool = False

    data_base_path: str = "./Who&When"
    output_dir: str = "./outputs"

    train_subsets: List[str] = field(default_factory=lambda: ["Hand-Crafted", "Algorithm-Generated"])
    test_subsets: List[str] = field(default_factory=lambda: ["Algorithm-Generated"])

    seed: int = 42

    epochs: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 16

    embedding_dim: int = 384
    hidden_dim: int = 384

    lambda_proto: float = 0.05
    alpha_l2: float = 5.0
    beta_cos: float = 2.0

    # Always evaluate with truncation (truncate=False removed).
    truncate_after_mistake: bool = True

    min_mistake_idx_for_supervision: int = 2

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

    supervised_loss_weight: float = 0.5
    weak_train_backbone: bool = True
    content_mask_token: str = "[MASK]"
    max_train_trajs: int = -1
    max_eval_trajs: int = -1
    score_running_eps: float = 1e-6
    norm_mode: str = "prefix_only"  # prefix_only|full|none
    norm_min_steps: int = 2
    use_ground_truth: bool = False

    def __post_init__(self):
        print(f"Using device: {self.device}")
        if self.device == "cpu" and self.precision == "fp16":
            self.precision = "fp32"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
