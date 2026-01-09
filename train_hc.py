import argparse
import os
import time

import train as base


def _save_json(obj, out_path):
    base._save_json(obj, out_path)


def main():
    parser = argparse.ArgumentParser(description="MASC reproduction (Hand-Crafted only)")
    parser.add_argument("--mode", choices=["audit", "unsup", "weak"], default="unsup")
    parser.add_argument("--ablate", choices=["none", "agent_only", "content_mask", "shuffle_steps", "cross_subset", "all"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--truncate_after_mistake", action="store_true")
    parser.add_argument("--no_truncate_after_mistake", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--llm_stub", action="store_true")
    parser.add_argument("--norm_mode", choices=["prefix_only", "full", "none"], default="prefix_only")
    parser.add_argument("--use_ground_truth", action="store_true")
    parser.add_argument("--hc_finetune_epochs", type=int, default=0)
    parser.add_argument("--hc_oversample", type=int, default=1)
    parser.add_argument("--alpha_l2", type=float, default=None)
    parser.add_argument("--beta_cos", type=float, default=None)
    parser.add_argument("--clean_ratio", type=float, default=1.0)
    parser.add_argument("--clean_epochs", type=int, default=0)
    parser.add_argument("--lambda_proto", type=float, default=None)
    parser.add_argument("--norm_min_steps", type=int, default=2)
    parser.add_argument("--clean_scope", choices=["all", "hc"], default="all")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = base.Config()
    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.train_subsets = ["Hand-Crafted", "Algorithm-Generated"]
    cfg.test_subsets = ["Hand-Crafted"]
    cfg.norm_mode = args.norm_mode
    if args.use_ground_truth:
        cfg.use_ground_truth = True
    if args.alpha_l2 is not None:
        cfg.alpha_l2 = args.alpha_l2
    if args.beta_cos is not None:
        cfg.beta_cos = args.beta_cos
    if args.lambda_proto is not None:
        cfg.lambda_proto = args.lambda_proto
    if args.norm_min_steps is not None:
        cfg.norm_min_steps = args.norm_min_steps
    if args.lr is not None:
        cfg.lr = args.lr
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

    base.set_seed(cfg.seed)

    if args.mode == "audit":
        audit = base.run_audit(cfg)
        out_path = os.path.join(cfg.output_dir, f"hc_audit_{int(time.time())}.json")
        _save_json(audit, out_path)
        print("[Audit] Summary:", audit.get("summary"))
        return

    print("Initializing Models...")
    encoder = base.ContextEncoder(cfg).to(cfg.device)
    masc = base.MASC(cfg).to(cfg.device)
    scorer = base.StepScorer().to(cfg.device)

    train_trajs = base.load_trajectories(cfg.data_base_path, cfg.train_subsets, cfg)
    test_trajs = base.load_trajectories(cfg.data_base_path, cfg.test_subsets, cfg)
    raw_train_trajs = list(train_trajs)

    base.print_basic_stats("TRAIN", train_trajs, cfg)
    base.print_basic_stats("TEST", test_trajs, cfg)

    if args.mode == "unsup":
        train_trajs, notes = base._prepare_unsup_training(train_trajs, cfg)
        if args.hc_oversample > 1:
            hc_trajs = [t for t in train_trajs if "Hand-Crafted" in (t.get("file_path") or "")]
            if hc_trajs:
                train_trajs = train_trajs + (hc_trajs * (args.hc_oversample - 1))
                notes.append(f"hc_oversample={args.hc_oversample}")
        print("[Train] Unsupervised policy:", "; ".join(notes))

        base.init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        base.train(train_trajs, encoder, masc, cfg)

        if args.clean_ratio < 1.0 and args.clean_epochs > 0:
            def mean_raw_score(tr):
                m = tr.get("mistake_idx", -1)
                if m is None or m < 0:
                    return None
                q = tr["query"]
                steps = tr["steps"]
                upto = min(m + 1, len(steps))
                hist_so_far = []
                xhat_hist_detached = []
                scores = []
                with base.torch.no_grad():
                    for t in range(upto):
                        agent_name, content = steps[t]
                        q_tilde, h_tilde = encoder(q, hist_so_far)
                        gt = encoder.step_gt(agent_name, content)
                        x_hat = masc(q_tilde, h_tilde)
                        p_ctx = masc.proto_ctx(xhat_hist_detached)
                        l2 = base.F.mse_loss(x_hat, gt)
                        dcos = 1 - base.F.cosine_similarity(x_hat, p_ctx).mean()
                        scores.append(float(cfg.alpha_l2 * l2 + cfg.beta_cos * dcos))
                        hist_so_far.append((agent_name, content))
                        xhat_hist_detached.append(x_hat.detach())
                if not scores:
                    return None
                return float(base.np.mean(scores))

            scored = []
            if args.clean_scope == "hc":
                candidates = [t for t in raw_train_trajs if t.get("subset") == "Hand-Crafted"]
            else:
                candidates = raw_train_trajs
            for tr in candidates:
                s = mean_raw_score(tr)
                if s is not None:
                    scored.append((s, tr))
            if scored:
                scored.sort(key=lambda x: x[0])
                keep = max(1, int(len(scored) * args.clean_ratio))
                clean_trajs = [tr for _, tr in scored[:keep]]
                clean_prefix = base.build_prefix_normal_trajectories(clean_trajs, cfg)
                if clean_prefix:
                    notes.append(f"clean_ratio={args.clean_ratio}")
                    notes.append(f"clean_prefix_trajs={len(clean_prefix)}")
                    orig_epochs = cfg.epochs
                    cfg.epochs = args.clean_epochs
                    base.train(clean_prefix, encoder, masc, cfg)
                    cfg.epochs = orig_epochs

        if args.hc_finetune_epochs > 0:
            hc_trajs = [t for t in raw_train_trajs if t.get("subset") == "Hand-Crafted"]
            if hc_trajs:
                hc_train_trajs, hc_notes = base._prepare_unsup_training(hc_trajs, cfg)
                print("[Train] HC finetune policy:", "; ".join(hc_notes))
                orig_epochs = cfg.epochs
                cfg.epochs = args.hc_finetune_epochs
                base.train(hc_train_trajs, encoder, masc, cfg)
                cfg.epochs = orig_epochs

        out = {"mode": "unsup", "notes": notes, "seed": cfg.seed, "subset": "Hand-Crafted", "norm_mode": cfg.norm_mode}
        for trunc in [True, False]:
            stats = base.evaluate_auc(test_trajs, encoder, masc, cfg, scorer=None, truncate_after_mistake=trunc)
            print(f"\n--- Evaluation (unsup, truncate={trunc}) ---")
            print(stats)
            out[f"eval_truncate_{trunc}"] = stats
        out_path = os.path.join(cfg.output_dir, f"hc_unsup_{int(time.time())}.json")
        _save_json(out, out_path)
        return

    if args.mode == "weak":
        base.init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
        base.train_weak(train_trajs, encoder, masc, scorer, cfg)

        out = {"mode": "weak", "seed": cfg.seed, "subset": "Hand-Crafted"}
        for trunc in [True, False]:
            stats = base.evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer, truncate_after_mistake=trunc)
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
                        train_trajs = base.load_trajectories(cfg.data_base_path, cfg.train_subsets, cfg)
                        test_trajs = base.load_trajectories(cfg.data_base_path, cfg.test_subsets, cfg)
                        base.init_prototype_simple(encoder, masc, train_trajs, max_steps=800)
                        base.train_weak(train_trajs, encoder, masc, scorer, cfg)
                        cross_key = f"{train_subset}_to_{test_subset}"
                        cross[cross_key] = base.evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer)
                    ablate_results[ab] = cross
                else:
                    ablate_results[ab] = base.evaluate_auc(test_trajs, encoder, masc, cfg, scorer=scorer, ablation=ab)

            out["ablation"] = ablate_results
            ab_path = os.path.join(cfg.output_dir, f"hc_ablation_{int(time.time())}.json")
            _save_json(out, ab_path)
        else:
            out_path = os.path.join(cfg.output_dir, f"hc_weak_{int(time.time())}.json")
            _save_json(out, out_path)


if __name__ == "__main__":
    main()
