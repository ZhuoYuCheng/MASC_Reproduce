import time

import train


def main():
    cfg = train.Config()
    audit = train.run_audit(cfg)
    out_path = f"{cfg.output_dir}/audit_{int(time.time())}.json"
    train._save_json(audit, out_path)
    print("[Audit] Summary:", audit.get("summary"))


if __name__ == "__main__":
    main()
