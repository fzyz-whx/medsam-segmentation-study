# -*- coding: utf-8 -*-
"""实验A：MedSAM 提示策略对比（零样本推理，四提示 × 三数据集测试集）。

推理流程严格复刻 MedSAM 官方仓库 bowang-lab/MedSAM 的 inference.py：
  图像 resize 1024x1024 (bicubic + antialias) → clamp/round 到 uint8 → min-max 归一化 →
  image_encoder → prompt_encoder(points=None, boxes=box_1024) → mask_decoder
  (multimask_output=False) → sigmoid(256x256) → 双线性插值回原始 H×W → 阈值 0.5。
  box_1024 = box / [W,H,W,H] * 1024（官方 stretch 缩放，非 letterbox）。

四种提示（setting）：
  box               GT 外接框（由二值 GT mask 计算，三数据集统一协议）
  box_perturb_10pct GT 外接框 ±10% 随机扰动（x/y 各坐标独立，幅度=U(-0.1,0.1)×框宽/高）
  center_point      GT 前景质心 → 以点为中心的伪框
  random_point      GT 前景内均匀随机点 → 伪框

点提示实现（如实记录）：MedSAM 官方推理代码只支持 box 提示（训练亦只用框），
故点提示转为伪框：以给定点为中心、半边长 = 0.05*max(H,W) 的正方形（即边长为
图像长边的 10%），clamp 到图像内，走与框提示完全相同的通路，不使用 GT 尺寸信息。

随机性：主种子 42；扰动框与随机点均由逐样本种子
random.Random("42|{dataset}|{setting}|{sample_id}") 生成，与处理顺序无关，可复现。

Dice 惯例：逐样本 Dice = 2|P∩G|/(|P|+|G|)；GT 与预测同为空记 1.0（两空一致）；
恰一方为空记 0.0；macro Dice = 逐样本 Dice 算术平均。三数据集 Phase 1 已验证无空 GT。

幂等：results/per_sample_A.csv 中已有的 (dataset, setting, sample_id) 行直接跳过；
聚合时由 per_sample_A.csv 全量重算，重写 metrics.csv 中 experiment=prompt_strategy
的行（保留其他实验的行），并更新 results/summary.json 的 prompt_strategy 键。

用法：
  .venv/Scripts/python.exe scripts/medsam_infer.py                  # 全量（建议后台跑）
  .venv/Scripts/python.exe scripts/medsam_infer.py --limit 2        # 试点（前2张/集）
  .venv/Scripts/python.exe scripts/medsam_infer.py --aggregate-only # 仅重算聚合
"""
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "medsam_vit_b.pth"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PER_SAMPLE_CSV = RESULTS / "per_sample_A.csv"
METRICS_CSV = RESULTS / "metrics.csv"
SUMMARY_JSON = RESULTS / "summary.json"
PREDS_DIR = RESULTS / "preds_A"
MEDSAM_VERIFY_JSON = RESULTS / "medsam_verify.json"

SEED = 42
MAX_TEST = 500  # 实验方案 5.3：每个数据集测试集最多 500 张
DATASET_ORDER = ["Kvasir-SEG", "ISIC2018", "Montgomery"]
SETTINGS = ["box", "box_perturb_10pct", "center_point", "random_point"]
PERTURB_FRAC = 0.10
PSEUDO_HALF_FRAC = 0.05
SAVE_MASKS_PER_DATASET = 20
PER_SAMPLE_COLS = ["experiment", "dataset", "setting", "sample_id", "dice",
                   "pred_empty", "gt_fg", "pred_fg"]
METRICS_COLS = ["experiment", "dataset", "setting", "n_samples", "dice"]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------- 数据与提示
def load_dataset_samples(name: str):
    """按 splits.json 取测试集样本（ISIC>500 时取 seed=42 洗牌顺序的前 500）。"""
    splits = json.loads((DATA / name / "splits.json").read_text(encoding="utf-8"))
    if name == "Kvasir-SEG":
        test = list(splits["test"])
        samples = [
            {"sample_id": f, "img": DATA / name / "images" / f, "mask": DATA / name / "masks" / f}
            for f in test
        ]
        info = f"test 全量 {len(test)} 张（≤500）"
    elif name == "ISIC2018":
        test = list(splits["test"])
        used = test[:MAX_TEST]
        samples = [
            {"sample_id": p["image"],
             "img": DATA / name / "images" / p["image"],
             "mask": DATA / name / "masks" / p["mask"]}
            for p in used
        ]
        info = f"test 共 {len(test)} 张，取 splits.json 顺序（seed=42 洗牌产物）前 {len(used)} 张"
    elif name == "Montgomery":
        test = list(splits["test"])
        samples = [
            {"sample_id": f"{cid}.png",
             "img": DATA / name / "images" / f"{cid}.png",
             "mask": DATA / name / "masks" / f"{cid}.png"}
            for cid in test
        ]
        info = f"test 全量 {len(test)} 张（≤500）"
    else:
        raise ValueError(f"未知数据集 {name}")
    return samples, info


def load_img_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)  # 灰度图统一转三通道


def load_gt(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        return np.array(m.convert("L")) >= 128  # Phase 1 约定：>=128 为前景


def sample_rng(dataset: str, setting: str, sample_id: str) -> random.Random:
    return random.Random(f"{SEED}|{dataset}|{setting}|{sample_id}")


def mask_bbox(gt: np.ndarray):
    ys, xs = np.nonzero(gt)
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def build_prompt_box(setting: str, dataset: str, sample_id: str, gt: np.ndarray, H: int, W: int):
    """返回 (x1,y1,x2,y2)，原始图像坐标系。"""
    x1, y1, x2, y2 = mask_bbox(gt)
    if setting == "box":
        return (x1, y1, x2, y2)
    if setting == "box_perturb_10pct":
        rng = sample_rng(dataset, setting, sample_id)
        w, h = x2 - x1, y2 - y1
        nx1 = x1 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC) * w
        nx2 = x2 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC) * w
        ny1 = y1 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC) * h
        ny2 = y2 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC) * h

        def fix(a: float, b: float, lo: float, hi: float):
            a = min(max(a, lo), hi - 1.0)
            b = min(max(b, lo + 1.0), hi)
            if b - a < 1.0:  # 扰动后退化则居中重设为 1px
                c = (a + b) / 2.0
                a = min(max(c - 0.5, lo), hi - 1.0)
                b = min(max(c + 0.5, lo + 1.0), hi)
            return a, b

        nx1, nx2 = fix(nx1, nx2, 0.0, float(W))
        ny1, ny2 = fix(ny1, ny2, 0.0, float(H))
        return (nx1, ny1, nx2, ny2)
    # ---- 点提示 → 伪框（官方推理代码仅支持 box 提示）----
    ys, xs = np.nonzero(gt)
    if setting == "center_point":
        cx, cy = float(xs.mean()), float(ys.mean())  # 前景质心（凹形/双肺并集可能落在前景外，如实保留）
    elif setting == "random_point":
        rng = sample_rng(dataset, setting, sample_id)
        i = rng.randrange(len(ys))
        cx, cy = float(xs[i]), float(ys[i])
    else:
        raise ValueError(setting)
    half = PSEUDO_HALF_FRAC * float(max(H, W))
    px1 = min(max(cx - half, 0.0), float(W - 1))
    py1 = min(max(cy - half, 0.0), float(H - 1))
    px2 = min(max(cx + half, px1 + 1.0), float(W))
    py2 = min(max(cy + half, py1 + 1.0), float(H))
    return (px1, py1, px2, py2)


def box_to_1024(box, W: int, H: int):
    x1, y1, x2, y2 = box
    return [x1 / W * 1024.0, y1 / H * 1024.0, x2 / W * 1024.0, y2 / H * 1024.0]


# ---------------------------------------------------------------- MedSAM 推理
def preprocess(img: np.ndarray, device) -> torch.Tensor:
    """复刻官方：resize 1024 bicubic+antialias → uint8 化 → min-max 归一化。"""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    x = F.interpolate(t, size=(1024, 1024), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp_(0, 255).round_().to(torch.uint8).float()  # 复刻官方 .astype(np.uint8)
    x = (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-8)
    return x.to(device)


@torch.no_grad()
def medsam_predict(model, img_embed: torch.Tensor, box_1024, H: int, W: int) -> np.ndarray:
    """官方 medsam_inference：box → prompt_encoder → mask_decoder → sigmoid → 回原尺寸。"""
    box_t = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device).reshape(1, 1, 4)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None, boxes=box_t, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    prob = torch.sigmoid(low_res_logits)  # (1,1,256,256)
    seg = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return seg.detach().cpu().numpy()


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    ps, gs = int(p.sum()), int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0  # 两空一致
    if ps == 0 or gs == 0:
        return 0.0
    inter = int(np.logical_and(p, g).sum())
    return 2.0 * inter / (ps + gs)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 1.0


# ---------------------------------------------------------------- 逐样本记录
def load_done() -> dict:
    """读取已有逐样本记录 {(dataset, setting, sample_id): row_dict}，重复行取后者。"""
    done: dict = {}
    if not PER_SAMPLE_CSV.exists():
        return done
    with open(PER_SAMPLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("sample_id"):
                continue
            try:
                key = (row["dataset"], row["setting"], row["sample_id"])
                row["dice"] = float(row["dice"])
                done[key] = row
            except (KeyError, ValueError):
                continue  # 跳过可能被截断的损坏行
    return done


def run_inference(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] device={device}", flush=True)
    from segment_anything import sam_model_registry
    print(f"[{ts()}] 加载 MedSAM 权重: {CKPT}", flush=True)
    model = sam_model_registry["vit_b"](checkpoint=str(CKPT))
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] 模型就绪：SAM ViT-B 结构，参数量 {n_params/1e6:.1f}M", flush=True)

    done = load_done()
    print(f"[{ts()}] 已有逐样本记录 {len(done)} 条（跳过）", flush=True)

    datasets = [d for d in (args.datasets.split(",") if args.datasets else DATASET_ORDER)]
    settings = [s for s in (args.settings.split(",") if args.settings else SETTINGS)]
    for d in datasets:
        if d not in DATASET_ORDER:
            raise SystemExit(f"未知数据集 {d}，可选 {DATASET_ORDER}")
    for s in settings:
        if s not in SETTINGS:
            raise SystemExit(f"未知 setting {s}，可选 {SETTINGS}")

    write_header = not PER_SAMPLE_CSV.exists()
    fout = open(PER_SAMPLE_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(PER_SAMPLE_COLS)

    failures = []
    t0 = time.time()
    n_img_done = 0
    try:
        for ds in datasets:
            samples, info = load_dataset_samples(ds)
            if args.limit:
                samples = samples[: args.limit]
            print(f"[{ts()}] === {ds}: 本轮 {len(samples)} 张（{info}）× 设置 {settings}", flush=True)
            for i, s in enumerate(samples):
                missing = [st for st in settings if (ds, st, s["sample_id"]) not in done]
                if not missing:
                    continue
                try:
                    img = load_img_rgb(s["img"])
                    gt = load_gt(s["mask"])
                except Exception as e:  # noqa: BLE001
                    msg = f"{ds}/{s['sample_id']}: 读取失败 {type(e).__name__}: {e}"
                    print(f"[{ts()}] [FAIL] {msg}", flush=True)
                    failures.append(msg)
                    continue
                H, W = gt.shape
                if int(gt.sum()) == 0:
                    msg = f"{ds}/{s['sample_id']}: GT 为空（Phase 1 约定下不应出现）"
                    print(f"[{ts()}] [FAIL] {msg}", flush=True)
                    failures.append(msg)
                    continue
                x = preprocess(img, device)
                with torch.no_grad():
                    embed = model.image_encoder(x)
                for st in missing:
                    box = build_prompt_box(st, ds, s["sample_id"], gt, H, W)
                    with torch.no_grad():
                        prob = medsam_predict(model, embed, box_to_1024(box, W, H), H, W)
                    pred = prob > 0.5
                    d = dice_score(pred, gt)
                    writer.writerow(["prompt_strategy", ds, st, s["sample_id"],
                                     f"{d:.6f}", int(pred.sum() == 0), int(gt.sum()), int(pred.sum())])
                    done[(ds, st, s["sample_id"])] = d
                    if args.limit:  # 试点模式打印明细便于人工核查
                        print(f"[{ts()}]   {ds}/{s['sample_id']:<35s} {st:<18s} dice={d:.4f} "
                              f"pred_fg={int(pred.sum())} gt_fg={int(gt.sum())}", flush=True)
                    if st == "box" and i < SAVE_MASKS_PER_DATASET:
                        outp = PREDS_DIR / ds / (Path(s["sample_id"]).stem + ".png")
                        if not outp.exists():
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(pred.astype(np.uint8) * 255).save(outp)
                fout.flush()
                n_img_done += 1
                if (i + 1) % 10 == 0 or i + 1 == len(samples):
                    rate = n_img_done / max(time.time() - t0, 1e-6)
                    print(f"[{ts()}] {ds}: {i + 1}/{len(samples)} 张处理完毕 "
                          f"(累计记录 {len(done)}，{rate:.2f} 张/s)", flush=True)
    finally:
        fout.close()
    print(f"[{ts()}] 推理循环结束：本轮处理 {n_img_done} 张，累计逐样本记录 {len(done)} 条，"
          f"用时 {time.time() - t0:.1f}s，失败 {len(failures)} 例", flush=True)
    for msg in failures:
        print(f"[{ts()}] [FAIL-SUMMARY] {msg}", flush=True)


# ---------------------------------------------------------------- 聚合
def kavsir_bbox_agreement() -> dict:
    """Kvasir mask 外接框 vs 数据集自带 kavsir_bboxes.json 的一致性（仅说明用）。"""
    bboxes = json.loads((DATA / "Kvasir-SEG" / "kavsir_bboxes.json").read_text(encoding="utf-8"))
    samples, _ = load_dataset_samples("Kvasir-SEG")
    ious = []
    for s in samples:
        stem = Path(s["sample_id"]).stem
        if stem not in bboxes or not bboxes[stem].get("bbox"):
            continue
        b = bboxes[stem]["bbox"][0]
        gt = load_gt(s["mask"])
        if int(gt.sum()) == 0:
            continue
        ious.append(box_iou(mask_bbox(gt), (b["xmin"], b["ymin"], b["xmax"], b["ymax"])))
    return {"n_compared": len(ious),
            "mean_box_iou": round(float(np.mean(ious)), 6) if ious else None,
            "min_box_iou": round(float(np.min(ious)), 6) if ious else None,
            "note": "实验A 的框提示统一由二值 GT mask 现算外接框（三数据集同协议）；"
                    "此检查说明其与 Kvasir 官方自带 kavsir_bboxes.json 基本一致"}


def aggregate() -> None:
    done = load_done()
    by_combo: dict = {}
    for (ds, st, sid), row in done.items():
        by_combo.setdefault((ds, st), []).append(row)

    metrics_rows = []
    results_tree: dict = {}
    incomplete = []
    for ds in DATASET_ORDER:
        samples, info = load_dataset_samples(ds)
        expected = len(samples)
        results_tree[ds] = {}
        for st in SETTINGS:
            recs = by_combo.get((ds, st), [])
            if len(recs) != expected:
                incomplete.append(f"{ds}×{st}: {len(recs)}/{expected}")
                continue
            dices = [r["dice"] for r in recs]
            n_pred_empty = sum(1 for r in recs if int(r.get("pred_empty", 0)) == 1)
            macro = float(np.mean(dices))
            metrics_rows.append({"experiment": "prompt_strategy", "dataset": ds,
                                 "setting": st, "n_samples": expected,
                                 "dice": f"{macro:.6f}"})
            results_tree[ds][st] = {"n_samples": expected, "macro_dice": round(macro, 6),
                                    "n_pred_empty": n_pred_empty}
    if incomplete:
        print(f"[{ts()}] [WARN] 以下组合样本数不齐，本轮不写 metrics（补跑后自动补齐）: "
              f"{'; '.join(incomplete)}", flush=True)

    # metrics.csv：保留其他实验的行，重写 prompt_strategy 的行
    keep = []
    if METRICS_CSV.exists():
        with open(METRICS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") != "prompt_strategy":
                    keep.append(row)
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_COLS)
        w.writeheader()
        for row in keep:
            w.writerow({k: row.get(k, "") for k in METRICS_COLS})
        for row in metrics_rows:
            w.writerow(row)
    print(f"[{ts()}] metrics.csv 已更新：prompt_strategy {len(metrics_rows)} 行"
          f"（保留其他实验行 {len(keep)} 行）", flush=True)

    # summary.json
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[{ts()}] [WARN] summary.json 读取失败（{e}），将重建", flush=True)
    model_info = {"name": "MedSAM (ViT-B)", "checkpoint": "checkpoints/medsam_vit_b.pth",
                  "source": "Zenodo 官方 DOI 存档 https://zenodo.org/records/10689643"}
    if MEDSAM_VERIFY_JSON.exists():
        try:
            model_info["verification"] = json.loads(MEDSAM_VERIFY_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    samples_ks, info_ks = load_dataset_samples("Kvasir-SEG")
    samples_is, info_is = load_dataset_samples("ISIC2018")
    samples_mt, info_mt = load_dataset_samples("Montgomery")
    summary["prompt_strategy"] = {
        "experiment": "实验A：MedSAM 零样本提示策略对比",
        "script": "scripts/medsam_infer.py",
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": model_info,
        "inference_protocol": {
            "pipeline": "复刻 MedSAM 官方 inference.py：resize 1024x1024 (bicubic+antialias) → "
                        "uint8 化 → min-max 归一化 → image_encoder → prompt_encoder(boxes) → "
                        "mask_decoder(multimask_output=False) → sigmoid → 双线性回原始 H×W → 0.5 阈值",
            "box_scaling": "box_1024 = box / [W,H,W,H] * 1024（官方 stretch 缩放，非 letterbox）",
            "segment_anything_pkg": "pip segment-anything==1.0（Meta 官方包，清华源安装）",
            "grayscale": "灰度图（Montgomery 胸片）统一 convert('RGB') 三通道输入",
        },
        "prompt_settings": {
            "box": "GT 外接框：由二值 GT mask 计算 [min_x, min_y, max_x+1, max_y+1]，三数据集统一协议",
            "box_perturb_10pct": "GT 外接框每坐标独立加 U(-0.1,0.1)×(框宽/高)，clamp 到图像内且保持≥1px",
            "center_point": "GT 前景质心（前景像素坐标均值；双肺并集等凹形目标可能落在前景外，如实保留）",
            "random_point": "GT 前景内均匀随机像素",
        },
        "point_prompt_implementation": (
            "MedSAM 官方推理代码仅支持 box 提示（训练亦只用框），故点提示按实验方案指示转为伪框："
            "以给定点为中心、半边长=0.05*max(H,W)（即边长为图像长边 10%）的正方形，clamp 到图像内，"
            "走与框提示完全相同的通路；不使用任何 GT 尺寸信息。"
        ),
        "randomness": (
            f"主种子 {SEED}；box_perturb_10pct 与 random_point 由逐样本种子 "
            "random.Random('42|{dataset}|{setting}|{sample_id}') 生成，与处理顺序无关，可复现"
        ),
        "dice_convention": (
            "逐样本 Dice=2|P∩G|/(|P|+|G|)；GT 与预测同为空记 1.0（两空一致）；恰一方为空记 0.0；"
            "macro Dice=逐样本 Dice 的算术平均。三数据集 Phase 1 已验证无空 GT，故实际仅预测可能为空（记 0）"
        ),
        "test_sets": {
            "Kvasir-SEG": {"n_test": len(json.loads((DATA / 'Kvasir-SEG' / 'splits.json').read_text(encoding='utf-8'))['test']),
                            "n_used": len(samples_ks), "sampling": info_ks},
            "ISIC2018": {"n_test": 519, "n_used": len(samples_is), "sampling": info_is},
            "Montgomery": {"n_test": 28, "n_used": len(samples_mt), "sampling": info_mt},
        },
        "kavsir_bbox_check": kavsir_bbox_agreement(),
        "results": results_tree,
        "per_sample_file": "results/per_sample_A.csv",
        "pred_masks": (
            f"每数据集前 {SAVE_MASKS_PER_DATASET} 张测试样本的 box 提示预测 mask，"
            "存 results/preds_A/<dataset>/<sample_stem>.png（0/255，原始分辨率）"
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ts()}] summary.json 已更新（prompt_strategy 键）", flush=True)

    # 控制台汇总表
    print(f"[{ts()}] ===== 实验 A 汇总（macro Dice）=====", flush=True)
    print(f"{'dataset':<12s} " + " ".join(f"{st:>18s}" for st in SETTINGS), flush=True)
    for ds in DATASET_ORDER:
        cells = []
        for st in SETTINGS:
            r = results_tree.get(ds, {}).get(st)
            cells.append(f"{r['macro_dice']:>18.6f}" if r else f"{'N/A':>18s}")
        print(f"{ds:<12s} " + " ".join(cells), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="实验A：MedSAM 提示策略对比")
    ap.add_argument("--datasets", type=str, default=None, help="逗号分隔：Kvasir-SEG,ISIC2018,Montgomery")
    ap.add_argument("--settings", type=str, default=None, help="逗号分隔：box,box_perturb_10pct,center_point,random_point")
    ap.add_argument("--limit", type=int, default=0, help="每数据集只处理前 N 张（试点用；0=全量）")
    ap.add_argument("--aggregate-only", action="store_true", help="只重算聚合，不推理")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    print(f"[{ts()}] 实验A 启动：python={'.'.join(map(str, sys.version_info[:3]))} "
          f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not args.aggregate_only:
        run_inference(args)
    aggregate()
    print(f"[{ts()}] 实验A 结束", flush=True)


if __name__ == "__main__":
    main()
