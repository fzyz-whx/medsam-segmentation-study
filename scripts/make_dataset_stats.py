# -*- coding: utf-8 -*-
"""Phase 4：生成 results/dataset_stats.csv（论文表 1「数据集统计」的唯一数据来源）。

数据来源（全部为 results/ 或 data/ 下的落盘文件，不凭空编写）：
  - 张数：data/<ds>/splits.json（Phase 1 落盘，seed=42，8:2）+ data/<ds>/stats.json（Phase 1 校验产物）
  - 分辨率范围：直接遍历 data/<ds>/images/ 用 PIL 读取图像头实测（min/max 宽×高）
  - 实验实际使用的测试张数：results/metrics.csv 中 experiment=prompt_strategy, setting=box 的
    n_samples（与实验 A/B/C 同一测试子集协议）

幂等：每次运行全量重算重写（读文件头很快），产物确定。
用法：.venv/Scripts/python.exe scripts/make_dataset_stats.py
"""
import csv
import json
import sys
from pathlib import Path

from PIL import Image

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "dataset_stats.csv"

DATASET_ORDER = ["Kvasir-SEG", "ISIC2018", "Montgomery"]

# 模态与来源描述（与 data/<ds>/stats.json、data/README.md 记录一致，人工撰写、逐字核对过）
META = {
    "Kvasir-SEG": {
        "modality": "彩色内镜（结直肠息肉）",
        "task": "胃肠息肉分割",
        "source": "官方数据集（datasets.simula.no/kvasir-seg），经 ModelScope OmniData/Kvasir-SEG 官方原包镜像下载（字节数与官方 zip 一致）",
    },
    "ISIC2018": {
        "modality": "皮肤镜（dermoscopy）",
        "task": "皮肤病变分割",
        "source": "ISIC 2018 Challenge Task 1 官方包（isic-challenge-data.s3.amazonaws.com），经 ModelScope OmniData/ISIC_2018_Task_1 镜像下载（sha256 与官方一致）",
    },
    "Montgomery": {
        "modality": "灰度胸部 X 光（CXR）",
        "task": "肺野分割（左右肺 mask 并集）",
        "source": "NLM Montgomery County CXR Set 官方包（data.lhncbc.nlm.nih.gov），经 hf-mirror 镜像下载（sha256 校验通过）",
    },
}

COLS = ["dataset", "modality", "task", "source", "n_pairs", "n_train", "n_test",
        "n_test_used_in_experiments", "min_width_px", "min_height_px",
        "max_width_px", "max_height_px", "resolution_range"]


def split_sizes(ds: str) -> dict:
    sp = json.loads((DATA / ds / "splits.json").read_text(encoding="utf-8"))
    return {"n_train": len(sp["train"]), "n_test": len(sp["test"])}


def measure_resolution(ds: str) -> dict:
    """遍历 images/ 实测每张图的 (W,H)，返回 min/max。PIL 只读文件头，速度快。"""
    sizes = []
    for p in sorted((DATA / ds / "images").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(p) as im:
            sizes.append(im.size)  # (W, H)
    if not sizes:
        raise SystemExit(f"{ds}: images/ 下没有可读图像")
    ws = [w for w, _ in sizes]
    hs = [h for _, h in sizes]
    return {"n_measured": len(sizes),
            "min_width_px": min(ws), "min_height_px": min(hs),
            "max_width_px": max(ws), "max_height_px": max(hs)}


def n_test_used(ds: str) -> int:
    """实验 A/B/C 实际使用的测试张数（metrics.csv prompt_strategy/box 行）。"""
    with open(RESULTS / "metrics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["experiment"] == "prompt_strategy" and row["dataset"] == ds
                    and row["setting"] == "box"):
                return int(row["n_samples"])
    raise SystemExit(f"metrics.csv 缺少 prompt_strategy/{ds}/box 行，无法确定 n_test_used")


def main() -> None:
    rows = []
    for ds in DATASET_ORDER:
        stats = json.loads((DATA / ds / "stats.json").read_text(encoding="utf-8"))
        meta = META[ds]
        sp = split_sizes(ds)
        res = measure_resolution(ds)
        used = n_test_used(ds)
        n_pairs = int(stats.get("pairs") or stats.get("verification", {}).get("pairs") or 0)
        if n_pairs != sp["n_train"] + sp["n_test"]:
            raise SystemExit(f"{ds}: pairs={n_pairs} != train+test={sp}")
        if res["n_measured"] != n_pairs:
            raise SystemExit(f"{ds}: 实测图像数 {res['n_measured']} != pairs {n_pairs}")
        rows.append({
            "dataset": ds,
            "modality": meta["modality"],
            "task": meta["task"],
            "source": meta["source"],
            "n_pairs": n_pairs,
            "n_train": sp["n_train"],
            "n_test": sp["n_test"],
            "n_test_used_in_experiments": used,
            "min_width_px": res["min_width_px"],
            "min_height_px": res["min_height_px"],
            "max_width_px": res["max_width_px"],
            "max_height_px": res["max_height_px"],
            "resolution_range": (f"W {res['min_width_px']}–{res['max_width_px']} px × "
                                 f"H {res['min_height_px']}–{res['max_height_px']} px"),
        })
        print(f"[dataset_stats] {ds}: pairs={n_pairs} train={sp['n_train']} test={sp['n_test']} "
              f"used_test={used} 分辨率 {rows[-1]['resolution_range']}", flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[dataset_stats] 已写 {OUT_CSV}（{len(rows)} 行）", flush=True)


if __name__ == "__main__":
    main()
