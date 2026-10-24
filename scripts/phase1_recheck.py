# -*- coding: utf-8 -*-
"""Phase 1 总验收复核（数据复核，幂等可重跑）。

对三个数据集逐项复核：
1. images/masks 文件数 == stats.json 声明数量
2. 图-mask 一一配对（同名，扩展名按各集约定）
3. splits.json：train/test 数量 == 声明；无交集；并集 == 全部样本；每条目指向的文件真实存在
4. 全量 PIL verify（可读性，不解码像素）
5. 抽样全解码：图-mask 同尺寸、mask 二值 {0,255}（>=128 二值化约定）
   - Kvasir-SEG / Montgomery：全量对数不大，全量全解码
   - ISIC2018：全量 verify + 抽样 60 对全解码（大图 6748x4499 全量解码代价高）
6. MedSAM 权重：文件大小 + torch.load(weights_only=True) + 键名前缀

输出：results/phase1_recheck.json
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42


def pair_spec(ds: str):
    """返回 (images_dir, masks_dir, list_samples) 其中 list_samples() -> [(img_path, msk_path, sample_id)]"""
    if ds == "Kvasir-SEG":
        img_dir, msk_dir = DATA / ds / "images", DATA / ds / "masks"

        def samples():
            out = []
            for p in sorted(img_dir.glob("*.jpg")):
                m = msk_dir / p.name
                out.append((p, m, p.stem))
            return out
        return img_dir, msk_dir, samples
    if ds == "ISIC2018":
        img_dir, msk_dir = DATA / ds / "images", DATA / ds / "masks"

        def samples():
            out = []
            for p in sorted(img_dir.glob("*.jpg")):
                m = msk_dir / (p.stem + "_segmentation.png")
                out.append((p, m, p.stem))
            return out
        return img_dir, msk_dir, samples
    if ds == "Montgomery":
        img_dir, msk_dir = DATA / ds / "images", DATA / ds / "masks"

        def samples():
            out = []
            for p in sorted(img_dir.glob("*.png")):
                m = msk_dir / p.name
                out.append((p, m, p.stem))
            return out
        return img_dir, msk_dir, samples
    raise ValueError(ds)


def split_ids(ds: str):
    d = json.loads((DATA / ds / "splits.json").read_text(encoding="utf-8"))
    if ds == "ISIC2018":
        tr = [x["image"].rsplit(".", 1)[0] for x in d["train"]]
        te = [x["image"].rsplit(".", 1)[0] for x in d["test"]]
    else:
        tr = [Path(x).stem if isinstance(x, str) else x for x in d["train"]]
        te = [Path(x).stem if isinstance(x, str) else x for x in d["test"]]
    return d, tr, te


def recheck_dataset(ds: str, decode_sample_n: int):
    stats = json.loads((DATA / ds / "stats.json").read_text(encoding="utf-8"))
    img_dir, msk_dir, samples_fn = pair_spec(ds)
    samples = samples_fn()

    rep = {"dataset": ds, "declared_pairs": stats.get("pairs")
           or stats.get("verification", {}).get("pairs")
           or stats.get("verification", {}).get("images_count")}

    # 1. counts
    img_files = [p for p in img_dir.iterdir() if p.is_file()]
    msk_files = [p for p in msk_dir.iterdir() if p.is_file()]
    rep["images_dir_count"] = len(img_files)
    rep["masks_dir_count"] = len(msk_files)
    rep["pairs_found"] = len(samples)
    rep["count_matches_declared"] = (len(samples) == rep["declared_pairs"])

    # 2. pairing: every image has its mask
    missing_mask = [s[2] for s in samples if not s[1].exists()]
    orphan_masks = [p.name for p in msk_files if p.suffix.lower() in (".png", ".jpg")
                    and not any(s[1] == p for s in samples)]
    rep["missing_mask"] = len(missing_mask)
    rep["orphan_mask_files"] = orphan_masks[:10]

    # 3. splits
    sd, tr, te = split_ids(ds)
    rep["split_train_declared"] = len(sd["train"]) if "train" in sd else sd.get("n_train")
    rep["split_test_declared"] = len(sd["test"]) if "test" in sd else sd.get("n_test")
    rep["split_train_actual"] = len(tr)
    rep["split_test_actual"] = len(te)
    rep["split_no_overlap"] = len(set(tr) & set(te)) == 0
    rep["split_covers_all"] = set(tr) | set(te) == {s[2] for s in samples}
    rep["split_seed"] = sd.get("seed")
    # 文件存在性统一核验（按各集 mask 命名）
    exist_ok = True
    id2sample = {s[2]: s for s in samples}
    for i in tr + te:
        if i not in id2sample:
            exist_ok = False
            break
        if not (id2sample[i][0].exists() and id2sample[i][1].exists()):
            exist_ok = False
            break
    rep["split_files_exist"] = exist_ok

    # 4. full verify (header decode)
    unread = []
    for ip, mp, sid in samples:
        try:
            with Image.open(ip) as im:
                im.verify()
            with Image.open(mp) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            unread.append((sid, repr(e)))
    rep["verify_unreadable"] = unread[:10]
    rep["verify_all_readable"] = len(unread) == 0

    # 5. decode sample: same size + mask binary
    rng = random.Random(SEED)
    if decode_sample_n >= len(samples):
        chosen = samples
        rep["decode_mode"] = f"full ({len(samples)} pairs)"
    else:
        chosen = rng.sample(samples, decode_sample_n)
        rep["decode_mode"] = f"sampled {len(chosen)}/{len(samples)} (seed={SEED})"
    size_mismatch, nonbinary, empty_after_bin = [], [], []
    mask_modes = {}
    midgray_max = 0.0
    for ip, mp, sid in chosen:
        with Image.open(ip) as im:
            im.load()
            isz = im.size
        with Image.open(mp) as im:
            im.load()
            mmode, msz = im.mode, im.size
            mask_modes[mmode] = mask_modes.get(mmode, 0) + 1
            if mmode != "L":
                im = im.convert("L")
            hist = im.histogram()
        if isz != msz:
            size_mismatch.append(sid)
        total = sum(hist)
        nz = [i for i, c in enumerate(hist) if c > 0]
        # L 模式严格二值 {0,255}；RGB(JPEG) 模式允许少量中间灰度伪影（>=128 二值化约定），
        # 记录最大中间灰度占比供审阅
        if mmode == "L" and not set(nz).issubset({0, 255}):
            nonbinary.append((sid, mmode, sorted(set(nz))[:8]))
        if mmode != "L":
            mid = sum(hist[1:255])
            midgray_max = max(midgray_max, mid / total if total else 0.0)
        if hist[255] == 0:  # >=128 二值化后前景为空
            empty_after_bin.append(sid)
    rep["decode_pairs"] = len(chosen)
    rep["decode_mask_modes"] = mask_modes
    rep["decode_midgray_max_ratio"] = round(midgray_max, 5)
    rep["decode_size_mismatch"] = size_mismatch[:10]
    rep["decode_nonbinary_mask"] = nonbinary[:10]
    rep["decode_empty_gt_after_bin"] = empty_after_bin[:10]
    rep["decode_all_ok"] = not size_mismatch and not nonbinary and not empty_after_bin
    return rep


def recheck_weights():
    ckpt = ROOT / "checkpoints" / "medsam_vit_b.pth"
    rep = {"path": str(ckpt), "exists": ckpt.exists()}
    if not rep["exists"]:
        rep["ok"] = False
        return rep
    rep["size_bytes"] = ckpt.stat().st_size
    rep["size_ok"] = rep["size_bytes"] == 375049145
    import torch
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    keys = list(sd.keys())
    rep["torch_load_ok"] = True
    rep["n_keys"] = len(keys)
    prefixes = sorted({k.split(".")[0] for k in keys})
    rep["prefixes"] = prefixes
    rep["prefix_ok"] = set(prefixes) == {"image_encoder", "prompt_encoder", "mask_decoder"}
    rep["first_key"] = keys[0]
    rep["ok"] = rep["size_ok"] and rep["torch_load_ok"] and rep["prefix_ok"] and rep["n_keys"] == 314
    return rep


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"generated_by": "scripts/phase1_recheck.py", "seed": SEED}
    report["Kvasir-SEG"] = recheck_dataset("Kvasir-SEG", decode_sample_n=10**9)   # full
    report["ISIC2018"] = recheck_dataset("ISIC2018", decode_sample_n=60)          # sampled
    report["Montgomery"] = recheck_dataset("Montgomery", decode_sample_n=10**9)   # full
    report["medsam_weights"] = recheck_weights()

    ok = True
    for ds in ("Kvasir-SEG", "ISIC2018", "Montgomery"):
        r = report[ds]
        ds_ok = (r["count_matches_declared"] and r["missing_mask"] == 0
                 and r["verify_all_readable"] and r["decode_all_ok"]
                 and r["split_no_overlap"] and r["split_covers_all"] and r["split_files_exist"]
                 and r["split_train_actual"] == r["split_train_declared"]
                 and r["split_test_actual"] == r["split_test_declared"])
        report[ds]["all_checks_passed"] = ds_ok
        ok = ok and ds_ok
    ok = ok and report["medsam_weights"]["ok"]
    report["phase1_acceptance"] = "PASS" if ok else "FAIL"

    out = RESULTS / "phase1_recheck.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v.get("all_checks_passed", v.get("ok")) if isinstance(v, dict) else v)
                      for k, v in report.items()}, ensure_ascii=False))
    print("acceptance:", report["phase1_acceptance"])
    print("report ->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
