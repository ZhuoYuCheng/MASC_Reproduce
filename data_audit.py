import time

from masc.config import Config
from masc.data import run_audit
from masc.utils import save_json


def main():
    cfg = Config()
    audit = run_audit(cfg)
    out_path = f"{cfg.output_dir}/audit_{int(time.time())}.json"
    save_json(audit, out_path)
    print("[Audit] Summary:", audit.get("summary"))


if __name__ == "__main__":
    main()
