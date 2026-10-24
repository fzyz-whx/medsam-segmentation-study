# -*- coding: utf-8 -*-
"""Phase 0 环境验收：CUDA 可用性 + GPU 小张量运算。幂等可重跑。"""
import json
import sys
from pathlib import Path

import torch

info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
}

if torch.cuda.is_available():
    info["device_name"] = torch.cuda.get_device_name(0)
    info["device_capability"] = list(torch.cuda.get_device_capability(0))
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).mean().item()
    info["gpu_matmul_mean"] = c  # 随机矩阵乘均值，非确定性数字，仅证明运算成功
    info["gpu_matmul_ok"] = True
    torch.cuda.synchronize()

out = Path("results/env_check.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False, indent=2))
ok = info["cuda_available"] and info.get("gpu_matmul_ok", False)
sys.exit(0 if ok else 1)
