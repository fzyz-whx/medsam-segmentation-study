# -*- coding: utf-8 -*-
"""
Montgomery County CXR Set (肺分割) 落袋与校验（幂等可重跑）。

步骤：
1) 校验 data/Montgomery/downloads/MontgomerySet.zip 大小 + ZIP CRC 完整性
2) 解压到 data/Montgomery/raw/（marker 文件保证幂等）
3) 构建配对结构：images/<case>.png（原图拷贝） + masks/<case>.png（左/右肺 mask 并集, 0/255）
4) 全量校验：PIL 逐张可读、图/mask 同尺寸、mask 二值、数量与官方声明(138)一致
5) stats.json：张数、尺寸分布、来源、下载方式、校验结论
6) splits.json：8:2 划分 seed=42（train/test 文件名列表，确定性可复现）
"""
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Montgomery"
ZIP_PATH = DATA / "downloads" / "MontgomerySet.zip"
RAW = DATA / "raw"
IMGS = DATA / "images"
MASKS = DATA / "masks"
STATS = DATA / "stats.json"
SPLITS = DATA / "splits.json"

EXPECTED_ZIP_SIZE = 616701553  # hf-mirror tree API 报告的字节数
OFFICIAL_COUNT = 138           # Jaeger et al. 2014 官方声明
SEED = 42
SOURCE_URL = ("https://hf-mirror.com/datasets/Famatsu123/"
              "montgomery-shenzhen-tuberculosis-cxr/resolve/main/MontgomerySet.zip")
ORIGIN_URL = ("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
              "Montgomery-County-CXR-Set/MontgomerySet/index.html")
CASE_RE = re.compile(r"^(MCUCXR_\d{4})")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def step_zip_check() -> dict:
    assert ZIP_PATH.exists(), f"zip 不存在: {ZIP_PATH}"
    size = ZIP_PATH.stat().st_size
    assert size == EXPECTED_ZIP_SIZE, f"zip 大小 {size} != 期望 {EXPECTED_ZIP_SIZE}"
    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        assert bad is None, f"ZIP CRC 校验失败: {bad}"
    return {"zip_size_bytes": size, "zip_sha256": sha256_of(ZIP_PATH),
            "zip_crc_ok": True}


def step_extract() -> Path:
    marker = RAW / ".extracted_ok"
    if marker.exists() and (RAW / "MontgomerySet").is_dir():
        print("[extract] 已存在, 跳过 (幂等)")
        return RAW / "MontgomerySet"
    if RAW.exists():
        shutil.rmtree(RAW)
    RAW.mkdir(parents=True)
    print("[extract] 解压中 ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW)
    setdir = RAW / "MontgomerySet"
    assert setdir.is_dir(), f"解压后未找到 {setdir}"
    marker.write_text("ok", encoding="utf-8")
    return setdir


def step_build_pairs(setdir: Path) -> list:
    """生成 images/ 与 masks/（左|右肺并集），幂等：输出已存在则跳过该 case。

    本镜像包的目录结构（与 NLM 原版一致, 仅 mask 目录名无下划线）:
        MontgomerySet/CXR_png/MCUCXR_XXXX_{0,1}.png   138 张 (+Thumbs.db 垃圾, glob *.png 天然排除)
        MontgomerySet/ManualMask/leftMask/<同名>.png   138 张
        MontgomerySet/ManualMask/rightMask/<同名>.png  138 张
    mask 文件名与原图完全同名, 按 stem 一一配对。
    """
    IMGS.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    cxr_dir = setdir / "CXR_png"
    left_dir = setdir / "ManualMask" / "leftMask"
    right_dir = setdir / "ManualMask" / "rightMask"
    cases = []
    for p in sorted(cxr_dir.glob("*.png")):
        assert CASE_RE.match(p.stem), f"无法解析 case id: {p.name}"
        stem = p.stem
        lp, rp = left_dir / p.name, right_dir / p.name
        assert lp.exists() and rp.exists(), f"case {stem} 缺少左右肺 mask ({lp.exists()=}/{rp.exists()=})"
        out_img = IMGS / f"{stem}.png"
        out_msk = MASKS / f"{stem}.png"
        if not out_img.exists():
            shutil.copyfile(p, out_img)  # 字节级拷贝, 无损
        if not out_msk.exists():
            l = np.array(Image.open(lp).convert("L"))
            r = np.array(Image.open(rp).convert("L"))
            assert l.shape == r.shape, f"case {stem} 左右 mask 尺寸不一致: {l.shape} vs {r.shape}"
            union = ((l > 127) | (r > 127)).astype(np.uint8) * 255
            Image.fromarray(union, mode="L").save(out_msk)
        cases.append(stem)
    assert len(cases) == len(set(cases)), "case stem 重复"
    return sorted(set(cases))


def step_verify(cases: list) -> dict:
    """全量校验：逐张 PIL 读取, 尺寸/模式/二值性, 图 mask 同尺寸。"""
    size_counter, mode_counter = Counter(), Counter()
    size_mismatch, unreadable, nonbinary = [], [], []
    pair_sizes = {}
    for cid in cases:
        ip, mp = IMGS / f"{cid}.png", MASKS / f"{cid}.png"
        try:
            im = Image.open(ip)
            im.verify()  # 解码完整性
            im = Image.open(ip)
            arr = np.array(im.convert("L"))
            mm = Image.open(mp)
            mm.verify()
            mm = Image.open(mp)
            marr = np.array(mm.convert("L"))
        except Exception as e:  # noqa: BLE001
            unreadable.append({"case": cid, "error": str(e)})
            continue
        wh = f"{im.width}x{im.height}"
        size_counter[wh] += 1
        mode_counter[im.mode] += 1
        if (im.width, im.height) != (mm.width, mm.height):
            size_mismatch.append({"case": cid, "img": [im.width, im.height],
                                  "mask": [mm.width, mm.height]})
        vals = np.unique(marr)
        if not np.isin(vals, [0, 255]).all():
            nonbinary.append({"case": cid, "values": vals.tolist()[:10]})
        pair_sizes[cid] = [im.width, im.height]
    n_img = len(list(IMGS.glob('*.png')))
    n_msk = len(list(MASKS.glob('*.png')))
    ok = (not unreadable and not size_mismatch and not nonbinary
          and len(cases) == OFFICIAL_COUNT and n_img == n_msk == OFFICIAL_COUNT)
    return {
        "pairs": len(cases),
        "images_count": n_img,
        "masks_count": n_msk,
        "official_declared_count": OFFICIAL_COUNT,
        "count_matches_official": len(cases) == OFFICIAL_COUNT,
        "image_size_distribution": dict(size_counter),
        "image_modes": dict(mode_counter),
        "image_mask_same_size": len(size_mismatch) == 0,
        "size_mismatch_cases": size_mismatch,
        "unreadable_cases": unreadable,
        "mask_nonbinary_cases": nonbinary,
        "all_checks_passed": ok,
        "pair_sizes": pair_sizes,
    }


def step_split(cases: list) -> dict:
    """8:2 划分, seed=42, 对排序后列表洗牌 => 确定性。幂等：重跑产出一致。"""
    cases = sorted(cases)
    rng = random.Random(SEED)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_train = int(round(0.8 * len(shuffled)))
    split = {
        "seed": SEED,
        "ratio": "8:2",
        "train": shuffled[:n_train],
        "test": shuffled[n_train:],
        "note": "对 sorted(case_id) 用 random.Random(42).shuffle 后前 80% 为 train; 幂等可复现",
    }
    assert len(split["train"]) + len(split["test"]) == len(cases)
    return split


def main() -> None:
    zipinfo = step_zip_check()
    print(f"[zip] OK size={zipinfo['zip_size_bytes']} sha256={zipinfo['zip_sha256'][:16]}...")
    setdir = step_extract()
    cases = step_build_pairs(setdir)
    print(f"[pairs] built {len(cases)} cases")
    ver = step_verify(cases)
    split = step_split(cases)
    SPLITS.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "dataset": "Montgomery",
        "full_name": "Montgomery County CXR Set (NLM, 胸片肺分割)",
        "reference": "Jaeger et al., 'Two public chest X-ray datasets for computer-aided screening', PMC4138378, 2014",
        "source_url": SOURCE_URL,
        "origin_url": ORIGIN_URL,
        "download_method": ("hf-mirror (HF_ENDPOINT=https://hf-mirror.com) 直下 Famatsu123/"
                            "montgomery-shenzhen-tuberculosis-cxr 仓库内的 MontgomerySet.zip "
                            "(NLM 原始目录结构镜像); GitHub release 无该数据集, ModelScope 无结果"),
        "zip_info": zipinfo,
        "structure": {
            "images": "data/Montgomery/images/<case>.png (原始 CXR, 字节级拷贝)",
            "masks": "data/Montgomery/masks/<case>.png (ManualMask leftMask|rightMask 并集, 0/255)",
            "raw": "data/Montgomery/raw/MontgomerySet/ (解压: CXR_png/ClinicalReadings/ManualMask{leftMask,rightMask})",
        },
        "verification": {k: v for k, v in ver.items() if k != "pair_sizes"},
        "size_policy": ("全部 138 对图/mask 同尺寸, 无需重采样对齐; 训练/推理时按模型输入统一 resize "
                        "(MedSAM: 1024x1024; U-Net: 256x256, mask 用最近邻插值)"),
        "splits_file": "data/Montgomery/splits.json",
        "split_sizes": {"train": len(split["train"]), "test": len(split["test"])},
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stats] all_checks_passed={ver['all_checks_passed']} "
          f"sizes={ver['image_size_distribution']} train={len(split['train'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
