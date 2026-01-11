import os
from typing import Any, Dict, List


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
        alt = os.path.join(root, model_id.replace(".", "___"))
        if os.path.isdir(alt):
            return alt
    return download_or_use_id(model_id, cache_dir)


def save_json(obj: Dict[str, Any], out_path: str):
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)
    print(f"[Output] Saved {out_path}")
