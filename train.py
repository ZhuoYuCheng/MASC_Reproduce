import argparse
import os
import time

from masc.config import Config, set_seed
from masc.data import load_trajectories, print_basic_stats, run_audit
from masc.eval_utils import evaluate_metrics
from masc.models import ContextEncoder, MASC, StepScorer
from masc.train_utils import init_prototype_simple, prepare_unsup_training, train_unsup, train_weak
from masc.utils import save_json


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
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--llm_stub", action="store_true")
    parser.add_argument("--norm_mode", choices=["prefix_only", "full", "none"], default="prefix_only")
    parser.add_argument("--norm_min_steps", type=int, default=2)
    parser.add_argument("--use_ground_truth", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha_l2", type=float, default=None)
    parser.add_argument("--beta_cos", type=float, default=None)
    parser.add_argument("--lambda_proto", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.train_subsets = [s.strip() for s in args.train_subsets.split(",") if s.strip()]
    cfg.test_subsets = [s.strip() for s in args.test_subsets.split(",") if s.strip()]
    cfg.norm_mode = args.norm_mode
    cfg.norm_min_steps = args.norm_min_steps
    if args.allow_download:
        cfg.local_files_only = False
    if args.local_files_only:
        cfg.local_files_only = True
    if args.llm_stub:
        cfg.llm_stub = True
    if args.use_ground_truth:
        cfg.use_ground_truth = True
    if args.lr is not None:
        cfg.lr = args.lr
    if args.alpha_l2 is not None:
        cfg.alpha_l2 = args.alpha_l2
    if args.beta_cos is not None:
        cfg.beta_cos = args.beta_cos
    if args.lambda_proto is not None:
        cfg.lambda_proto = args.lambda_proto

    set_seed(cfg.seed)

    if args.mode == "audit":
        audit = run_audit(cfg)
        out_path = os.path.join(cfg.output_dir, f"audit_{int(time.time())}.json")
        save_json(audit, out_path)
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
        train_trajs, notes = prepare_unsup_training(train_trajs, cfg)
        print("[Train] Unsupervised policy:", "; ".join(notes))

        init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        train_unsup(train_trajs, encoder, masc, cfg)

        out = {"mode": "unsup", "notes": notes, "seed": cfg.seed}
        stats = evaluate_metrics(test_trajs, encoder, masc, cfg, scorer=None)
        print("\n--- Evaluation (unsup, truncate=True) ---")
        print(stats)
        out["eval"] = stats
        out_path = os.path.join(cfg.output_dir, f"unsup_{int(time.time())}.json")
        save_json(out, out_path)
        return

    if args.mode == "weak":
        init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        train_weak(train_trajs, encoder, masc, scorer, cfg)

        out = {"mode": "weak", "seed": cfg.seed}
        stats = evaluate_metrics(test_trajs, encoder, masc, cfg, scorer=scorer)
        print("\n--- Evaluation (weak, truncate=True) ---")
        print(stats)
        out["eval"] = stats

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
                        cross[cross_key] = evaluate_metrics(test_trajs, encoder, masc, cfg, scorer=scorer)
                    ablate_results[ab] = cross
                else:
                    ablate_results[ab] = evaluate_metrics(test_trajs, encoder, masc, cfg, scorer=scorer, ablation=ab)

            out["ablation"] = ablate_results
            ab_path = os.path.join(cfg.output_dir, f"ablation_{int(time.time())}.json")
            save_json(out, ab_path)
        else:
            out_path = os.path.join(cfg.output_dir, f"weak_{int(time.time())}.json")
            save_json(out, out_path)
        return


if __name__ == "__main__":
    main()
