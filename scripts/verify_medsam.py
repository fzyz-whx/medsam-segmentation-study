# -*- coding: utf-8 -*-
"""MedSAM / SAM 权重校验：torch.load 加载 + state_dict 键名检查。
用法: python scripts/verify_medsam.py [checkpoints/medsam_vit_b.pth]
输出: results/medsam_verify.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "checkpoints" / "medsam_vit_b.pth"
OUT_JSON = ROOT / "results" / "medsam_verify.json"


def main() -> int:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    report = {
        "path": str(ckpt_path),
        "exists": ckpt_path.exists(),
        "size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else 0,
        "torch_load_ok": False,
        "n_keys": 0,
        "first_keys": [],
        "prefix_ok": False,
        "error": None,
    }
    if not ckpt_path.exists():
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        # 老格式权重可能 weights_only=True 失败，退回普通加载
        try:
            sd = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as e2:  # noqa: BLE001
            report["error"] = f"weights_only err: {e} | plain err: {e2}"
            OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1

    keys = list(sd.keys()) if hasattr(sd, "keys") else []
    report["torch_load_ok"] = True
    report["n_keys"] = len(keys)
    report["first_keys"] = keys[:8]
    # MedSAM/SAM 官方格式: image_encoder.* / prompt_encoder.* / mask_decoder.*
    prefixes = {"image_encoder", "prompt_encoder", "mask_decoder"}
    report["prefix_ok"] = bool(keys) and any(k.split(".")[0] in prefixes for k in keys)
    report["all_key_prefixes"] = sorted({k.split(".")[0] for k in keys})
    report["loaded_object_type"] = type(sd).__name__

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["prefix_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
