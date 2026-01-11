import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import resolve_model_dir


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
        q = self.Wq(p)
        k = self.Wk(X)
        v = self.Wv(X)
        att = torch.softmax((q @ k.T) / (p.size(-1) ** 0.5), dim=-1)
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
