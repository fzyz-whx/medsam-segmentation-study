# -*- coding: utf-8 -*-
"""Kvasir-SEG 数据集校验 + 8:2 划分 + 统计落盘（幂等可重跑）。

用法: .venv/Scripts/python.exe scripts/prepare_kvasir_seg.py
产物:
  data/Kvasir-SEG/splits.json  (train/test 文件名列表, seed=42, 可复现)
  data/Kvasir-SEG/stats.json   (张数/尺寸分布/来源/校验结论)
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# 注意：cv2.imread 在 Windows 中文路径（本仓库目录含中文）下会全部失败，
# 故统一用 PIL 读取 + numpy 转数组做数值分析；后续阶段需要 cv2 时必须用
# cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag) 规避。

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Kvasir-SEG"
IMG_DIR = DATA / "images"
MSK_DIR = DATA / "masks"
SEED = 42
OFFICIAL_PAIRS = 1000  # Jha et al. 2020 / datasets.simula.no 官方声明

SOURCES = {
    "official_page": "https://datasets.simula.no/kvasir-seg/",
    "official_zip": "https://datasets.simula.no/downloads/kvasir-seg.zip",
    "downloaded_from": (
        "ModelScope 镜像 OmniData/Kvasir-SEG "
        "https://modelscope.cn/datasets/OmniData/Kvasir-SEG/resolve/master/raw/kvasir-seg.zip"
    ),
    "download_method": (
        "curl 直连 ModelScope（国内直连，6.8MB/s，44MB）；"
        "该 zip 为官方 kvasir-seg.zip 的原包镜像（字节数一致 46227172）。"
        "GitHub simula/kvasir-seg 已迁移至 DebeshJha/Kvasir-SEG 且无 release，"
        "hf-mirror 搜索 kvasir 无结果，故按镜像优先级取 ModelScope。"
    ),
}


def main() -> None:
    img_names = sorted(p.name for p in IMG_DIR.glob("*.jpg"))
    msk_names = sorted(p.name for p in MSK_DIR.glob("*.jpg"))
    img_set, msk_set = set(img_names), set(msk_names)
    only_img = sorted(img_set - msk_set)
    only_msk = sorted(msk_set - img_set)
    pairs = sorted(img_set & msk_set)

    size_counter: Counter = Counter()
    size_mismatch = []
    unreadable = []
    mask_value_ranges = Counter()  # 记录 mask 数值域
    empty_gt = []  # >=128 二值化后前景为空的 GT
    inter_fracs = []  # 中间灰度像素占比（JPEG 边界伪影）

    for name in pairs:
        try:
            with Image.open(IMG_DIR / name) as im:
                im.load()
                w, h = im.size
                im_mode = im.mode
            with Image.open(MSK_DIR / name) as mk:
                mk.load()
                mw, mh = mk.size
            size_counter[f"{w}x{h}"] += 1
            if (w, h) != (mw, mh):
                size_mismatch.append(name)
            # mask 数值域检查（PIL→numpy；cv2.imread 中文路径不可用）
            m = np.array(mk.convert("L"))
            lo, hi = int(m.min()), int(m.max())
            mask_value_ranges[f"[{lo},{hi}]"] += 1
            inter_fracs.append(float(((m > 0) & (m < 255)).mean()))
            bin_m = m >= 128  # 二值化约定：>=128 为前景
            if bin_m.sum() == 0:
                empty_gt.append(name)
            _ = im_mode
        except Exception as e:  # noqa: BLE001
            unreadable.append({"name": name, "error": str(e)})

    # 8:2 划分（幂等：排序 + 固定种子）
    rng = random.Random(SEED)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * 0.8))
    splits = {
        "dataset": "Kvasir-SEG",
        "seed": SEED,
        "ratio": "8:2",
        "train": shuffled[:n_train],
        "test": shuffled[n_train:],
    }

    all_same_size = len(size_counter) == 1
    checks = {
        "count_matches_official": len(pairs) == OFFICIAL_PAIRS,
        "all_readable": len(unreadable) == 0,
        "image_mask_paired_1to1": not only_img and not only_msk,
        "image_mask_same_size": len(size_mismatch) == 0,
        "mask_value_range_0_255_all_pairs": set(mask_value_ranges) == {"[0,255]"},
        "jpeg_artifact_pixels_negligible": float(np.mean(inter_fracs)) < 0.025,
        "no_empty_gt_after_binarize_128": len(empty_gt) == 0,
    }
    uniform_resize_plan = (
        None
        if all_same_size
        else "尺寸不一致：训练/推理前统一 resize 到 1024x1024（MedSAM 输入）与 256x256（U-Net 输入），box/mask 坐标同步缩放"
    )

    stats = {
        "dataset": "Kvasir-SEG",
        "task": "胃肠息肉分割（polyp segmentation）",
        "pairs": len(pairs),
        "official_declared_pairs": OFFICIAL_PAIRS,
        "images_dir": "data/Kvasir-SEG/images",
        "masks_dir": "data/Kvasir-SEG/masks",
        "extra_files": ["kavsir_bboxes.json (GT 外接框, MedSAM 框提示可用)"],
        "size_distribution": dict(size_counter.most_common()),
        "uniform_size": all_same_size,
        "resize_plan_if_needed": uniform_resize_plan,
        "mask_value_ranges": dict(mask_value_ranges.most_common(10)),
        "mask_binarize_convention": ">=128 为前景（JPEG 边界伪影：中间灰度像素占比 mean=%.4f max=%.4f）"
        % (float(np.mean(inter_fracs)), float(np.max(inter_fracs))),
        "empty_gt_after_binarize": empty_gt,
        "size_mismatch_count": len(size_mismatch),
        "unreadable_count": len(unreadable),
        "only_in_images": only_img,
        "only_in_masks": only_msk,
        "checks": checks,
        "validation_conclusion": "PASS" if all(checks.values()) else "FAIL",
        "source": SOURCES,
        "split": {"seed": SEED, "ratio": "8:2", "train": len(splits["train"]), "test": len(splits["test"])},
    }

    (DATA / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print("pairs:", len(pairs))
    print("size_distribution:", dict(size_counter.most_common(10)))
    print("checks:", checks)
    print("split:", len(splits["train"]), "train /", len(splits["test"]), "test")
    print("validation_conclusion:", stats["validation_conclusion"])


if __name__ == "__main__":
    main()
