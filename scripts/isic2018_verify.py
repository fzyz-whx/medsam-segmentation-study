# -*- coding: utf-8 -*-
"""ISIC 2018 Task1 dataset verification, 8:2 split (seed=42), stats.json writer.

Idempotent: safe to re-run; overwrites stats.json / splits.json deterministically.
Layout after extraction/normalization (auto-detected, with fallbacks):
  data/ISIC2018/images/*.jpg   (official ISIC2018_Task1-2_Training_Input)
  data/ISIC2018/masks/*.png    (official ISIC2018_Task1_Training_GroundTruth, *_segmentation.png)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "ISIC2018"
SEED = 42
TRAIN_RATIO = 0.8
OFFICIAL_COUNT = 2594  # ISIC 2018 Task1-2 Training set official declaration


def find_dirs():
    """Locate image dir (jpg) and mask dir (png with _segmentation suffix)."""
    img_dir = DS / "images"
    mask_dir = DS / "masks"
    if img_dir.is_dir() and mask_dir.is_dir():
        return img_dir, mask_dir
    # fallback: search two levels deep for dirs matching official names
    for d in DS.rglob("*"):
        if d.is_dir():
            n = d.name.lower()
            if "training_input" in n and any(d.glob("*.jpg")):
                img_dir = d
            if "training_groundtruth" in n and any(d.glob("*.png")):
                mask_dir = d
    return img_dir, mask_dir


def main():
    img_dir, mask_dir = find_dirs()
    print(f"image dir: {img_dir}")
    print(f"mask  dir: {mask_dir}")
    if not (img_dir.is_dir() and mask_dir.is_dir()):
        print("FATAL: image/mask dirs not found")
        sys.exit(1)

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    masks = sorted(mask_dir.glob("*.png"))

    # pair by stem: image ISIC_xxxxxxx, mask ISIC_xxxxxxx_segmentation
    mask_map = {}
    for m in masks:
        stem = m.stem
        if stem.endswith("_segmentation"):
            stem = stem[: -len("_segmentation")]
        mask_map[stem] = m

    pairs, missing_mask, size_mismatch, unreadable = [], [], [], []
    size_dist = Counter()
    mask_modes = Counter()
    img_sizes, mask_sizes = [], []

    for p in imgs:
        stem = p.stem
        m = mask_map.get(stem)
        if m is None:
            missing_mask.append(p.name)
            continue
        try:
            with Image.open(p) as im:
                im.load()
                wi, hi = im.size
                img_sizes.append((wi, hi))
            with Image.open(m) as mk:
                mk.load()
                wm, hm = mk.size
                mask_modes[mk.mode] += 1
        except Exception as e:  # noqa: BLE001
            unreadable.append((p.name, str(e)))
            continue
        mask_sizes.append((wm, hm))
        size_dist[f"{wi}x{hi}"] += 1
        if (wi, hi) != (wm, hm):
            size_mismatch.append({"image": p.name, "image_size": [wi, hi], "mask_size": [wm, hm]})
        pairs.append((p, m))

    # mask value check on a sample (binary 0/255 expected)
    val_counter = Counter()
    for p, m in pairs[:200]:
        arr = np.array(Image.open(m))
        vals = np.unique(arr)
        val_counter[tuple(int(v) for v in vals[:5])] += 1

    n = len(pairs)
    print(f"pairs={n} missing_mask={len(missing_mask)} unreadable={len(unreadable)} "
          f"size_mismatch={len(size_mismatch)}")
    print(f"distinct image sizes: {len(size_dist)}; top5: {size_dist.most_common(5)}")
    print(f"mask modes: {mask_modes}; mask value sets (sample): {val_counter.most_common(5)}")

    # 8:2 split, seed 42, deterministic
    stems = sorted(pm[0].stem for pm in pairs)
    rng = random.Random(SEED)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    train, test = shuffled[:n_train], shuffled[n_train:]

    splits = {
        "dataset": "ISIC2018_Task1",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "n_train": len(train),
        "n_test": len(test),
        "train": [{"image": f"{s}.jpg", "mask": f"{s}_segmentation.png"} for s in train],
        "test": [{"image": f"{s}.jpg", "mask": f"{s}_segmentation.png"} for s in test],
    }
    (DS / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = (n == OFFICIAL_COUNT and not missing_mask and not unreadable)
    stats = {
        "dataset": "ISIC2018_Task1",
        "task": "skin lesion segmentation (Task 1)",
        "source": {
            "mirror": "ModelScope dataset OmniData/ISIC_2018_Task_1 (mirrors the official ISIC 2018 Task 1 zips verbatim)",
            "url": "https://modelscope.cn/datasets/OmniData/ISIC_2018_Task_1",
            "file": "raw/ISIC_2018_Task_1.tar.gz",
            "direct_download_url": "https://modelscope.cn/datasets/OmniData/ISIC_2018_Task_1/resolve/master/raw/ISIC_2018_Task_1.tar.gz",
            "archive_sha256": "506bacda9e63668268ab1727665563c1de85504095fca7c99dc2da3cee5eb87a",
            "archive_bytes": 13799882093,
            "official_origin": "ISIC 2018 challenge (challenge.isic-archive.com), data hosted at isic-challenge-data.s3.amazonaws.com/2018/",
            "download_method": "curl 8-way segmented range download with resume (scripts/isic2018_download_seg.sh); sha256 verified equal to ModelScope API declaration",
            "mirror_selection": "hf-mirror searched first per task rules: candidates zs389/isic2018Task1-2Training & cs198-cvmig & surajbijjahalli are classification-format parquets (no masks); MedOtter/ISIC2018 is full-res parquet with image/mask but hf-mirror gave 284KB/s (26h ETA); fell back to ModelScope official archive at 6-8MB/s",
        },
        "normalization": {
            "resize_applied": "none (raw official resolution kept on disk)",
            "resize_plan": "per-model resize at experiment time: 1024x1024 for MedSAM inference, 256x256 for U-Net training; images and masks resized with identical geometry",
        },
        "pairs": n,
        "official_declared_training_pairs": OFFICIAL_COUNT,
        "count_matches_official": n == OFFICIAL_COUNT,
        "images_dir": str(img_dir.relative_to(ROOT)),
        "masks_dir": str(mask_dir.relative_to(ROOT)),
        "missing_masks": missing_mask[:20],
        "unreadable": unreadable[:20],
        "size_mismatch_count": len(size_mismatch),
        "size_mismatch_examples": size_mismatch[:10],
        "distinct_image_sizes": len(size_dist),
        "top_image_sizes": size_dist.most_common(10),
        "image_wh_min": [int(min(s[0] for s in img_sizes)), int(min(s[1] for s in img_sizes))] if img_sizes else None,
        "image_wh_max": [int(max(s[0] for s in img_sizes)), int(max(s[1] for s in img_sizes))] if img_sizes else None,
        "mask_modes": dict(mask_modes),
        "mask_value_sets_sample": {str(k): v for k, v in val_counter.most_common(5)},
        "split": {"seed": SEED, "ratio": "8:2", "train": len(train), "test": len(test)},
        "verification_passed": ok and len(size_mismatch) == 0,
    }
    (DS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"splits.json + stats.json written. verification_passed={stats['verification_passed']}")
    return 0 if stats["verification_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
