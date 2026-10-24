# -*- coding: utf-8 -*-
"""实验C：跨数据集泛化（实验方案 5.3 实验 C，可选项；前置：实验 A/B 已完成）。

问题：在甲数据集上训练的专用 U-Net（full 档位，n=全部训练池）直接测乙数据集
（不同模态/域）测试集，macro Dice 相对乙数据集上的 MedSAM 零样本（GT 框提示）
如何衰减？——展示基础模型的跨域泛化优势。

组合（STATE.md Phase 1 建议，已核对 checkpoints 就绪）：
  U-Net(Kvasir-SEG, n=720) -> ISIC2018   test 500   彩色内镜 -> 皮肤镜（近域跨集）
  U-Net(Kvasir-SEG, n=720) -> Montgomery test 28    彩色内镜 -> 灰度胸片（最大模态差）
  U-Net(ISIC2018,  n=1868) -> Kvasir-SEG test 200   皮肤镜   -> 彩色内镜
  U-Net(ISIC2018,  n=1868) -> Montgomery test 28    皮肤镜   -> 灰度胸片
对照：MedSAM 零样本(乙) = 实验 A prompt_strategy/box 的**同测试集**数字，直接
复用不重推（脚本校验其存在于 metrics.csv 且 n_samples 与本实验测试子集一致；
若缺失则报错要求补测，绝不静默编造）。

推理协议：严格遵循 checkpoints/unet_<ds>_<n>.pth 内 meta 的预处理规则
（img_size=256、PIL 双线性、x/255；GT mask convert('L') + >=128 二值化）；
U-Net 概率图（256x256）双线性回原始分辨率后 0.5 阈值（与实验 A/B 对齐）；
Dice 惯例与实验 A/B 完全一致：两空=1.0、恰一空=0.0、macro=算术平均。
测试子集与实验 A/B 同一协议：Kvasir-SEG test 200 全量、ISIC2018 test 前 500/519、
Montgomery test 28 全量（splits.json，seed=42 划分）。

模型与工具函数复用 scripts/unet.py（UNet/dice_np/autocast_ctx/load_splits），
保证架构与指标口径逐字节一致。

幂等：某组合 per_sample_C.csv 行数 == 目标测试集样本数 且 marker 存在 → 跳过
推理；行数不齐则先删该组合旧行重测；metrics.csv/summary.json 的 cross_dataset
部分每次运行重写（保留其他实验行）。--force 强制全部重做。

用法：
  .venv/Scripts/python.exe scripts/cross_dataset.py --probe   # 试点：每组合前4张，不写任何结果
  .venv/Scripts/python.exe scripts/cross_dataset.py           # 全量（幂等）
  .venv/Scripts/python.exe scripts/cross_dataset.py --force   # 强制重跑全部组合
日志：logs/cross_dataset.log（建议后台运行）
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet import UNet, autocast_ctx, dice_np, load_splits, set_seed  # noqa: E402

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
MARKER_DIR = RESULTS / "cross_runs"
PER_SAMPLE_CSV = RESULTS / "per_sample_C.csv"
METRICS_CSV = RESULTS / "metrics.csv"
SUMMARY_JSON = RESULTS / "summary.json"

SEED = 42
MAX_TEST = 500  # 与实验 A/B 一致：每数据集测试集最多 500 张
# (源数据集, 目标数据集)：源 = 有 full 档位 U-Net checkpoint 的数据集，
# 目标测试子集与实验 A 完全一致（保证 MedSAM 数字可直接复用）
PAIRS = [
    ("Kvasir-SEG", "ISIC2018"),
    ("Kvasir-SEG", "Montgomery"),
    ("ISIC2018", "Kvasir-SEG"),
    ("ISIC2018", "Montgomery"),
]
MEDSAM_TARGETS = ["Kvasir-SEG", "ISIC2018", "Montgomery"]  # MedSAM 参考行顺序
VAL_DENOM = 10  # 与实验 B 一致：训练集末 1/10 为验证集，其余为训练池(full 档位 n)

try:
    _RS = Image.Resampling
except AttributeError:  # Pillow < 9.1
    _RS = Image
BILINEAR = _RS.BILINEAR
PER_SAMPLE_COLS = ["experiment", "dataset", "setting", "sample_id", "dice",
                   "pred_empty", "gt_fg", "pred_fg"]
METRICS_COLS = ["experiment", "dataset", "setting", "n_samples", "dice"]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------- 数据
def test_items(ds: str) -> list:
    """测试集样本，与实验 A/B 完全同一协议（ISIC 取前 500/519）。支持三个数据集。"""
    sp = load_splits(ds)
    test = sp["test"]
    if ds == "Kvasir-SEG":
        return [{"sample_id": e, "img": DATA / ds / "images" / e,
                 "mask": DATA / ds / "masks" / e} for e in test]
    if ds == "ISIC2018":
        return [{"sample_id": e["image"], "img": DATA / ds / "images" / e["image"],
                 "mask": DATA / ds / "masks" / e["mask"]} for e in test[:MAX_TEST]]
    if ds == "Montgomery":
        return [{"sample_id": e, "img": DATA / ds / "images" / f"{e}.png",
                 "mask": DATA / ds / "masks" / f"{e}.png"} for e in test]
    raise ValueError(f"未知数据集 {ds}")


def full_pool_size(ds: str) -> int:
    """full 档位训练池大小（与实验 B resolve_tier 同口径）：train 去掉末 1/10。"""
    n_train = len(load_splits(ds)["train"])
    return n_train - n_train // VAL_DENOM


def source_ckpt_path(ds: str) -> Path:
    n = full_pool_size(ds)
    p = CKPT_DIR / f"unet_{ds}_{n}.pth"
    if not p.exists():
        raise SystemExit(f"[{ts()}] 缺少 full 档位 checkpoint：{p}")
    return p


# ---------------------------------------------------------------- 逐样本记录 IO
def read_rows() -> list:
    if not PER_SAMPLE_CSV.exists():
        return []
    with open(PER_SAMPLE_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("sample_id")]


def count_rows(ds: str, setting: str) -> int:
    return sum(1 for r in read_rows() if r["dataset"] == ds and r["setting"] == setting)


def drop_rows(ds: str, setting: str) -> None:
    rows = read_rows()
    keep = [r for r in rows if not (r["dataset"] == ds and r["setting"] == setting)]
    if len(keep) == len(rows):
        return
    with open(PER_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SAMPLE_COLS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in PER_SAMPLE_COLS})
    print(f"[{ts()}]   清理 {ds} x {setting} 旧逐样本记录 {len(rows)-len(keep)} 行", flush=True)


def append_rows(rows: list) -> None:
    write_header = not PER_SAMPLE_CSV.exists()
    with open(PER_SAMPLE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SAMPLE_COLS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- 跨集推理
@torch.no_grad()
def evaluate_cross(src: str, tgt: str, model, img_size: int, device,
                   limit: int = 0, batch: int = 32) -> list:
    """甲模型在乙测试集逐样本评估：协议与实验 B evaluate_test 逐行一致，仅
    experiment/dataset/setting 标注不同。"""
    samples = test_items(tgt)
    if limit:
        samples = samples[:limit]
    model.eval()
    setting = f"UNet({src}→{tgt})"
    rows = []
    t0 = time.time()
    for i0 in range(0, len(samples), batch):
        chunk = samples[i0:i0 + batch]
        xs, gts = [], []
        for s in chunk:
            with Image.open(s["img"]) as im:
                im = im.convert("RGB").resize((img_size, img_size), BILINEAR)
                xs.append(np.array(im, dtype=np.uint8))
            with Image.open(s["mask"]) as m:
                gts.append(np.array(m.convert("L")) >= 128)
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(device).float() / 255.0
        with autocast_ctx(device):
            lg = model(x).squeeze(1)
        prob = torch.sigmoid(lg.float()).cpu()  # (B,img,img)
        for j, s in enumerate(chunk):
            H, W = gts[j].shape
            p = (F.interpolate(prob[j:j + 1, None], size=(H, W), mode="bilinear",
                               align_corners=False)[0, 0].numpy() > 0.5)
            d = dice_np(p, gts[j])
            rows.append({"experiment": "cross_dataset", "dataset": tgt, "setting": setting,
                         "sample_id": s["sample_id"], "dice": f"{d:.6f}",
                         "pred_empty": int(p.sum() == 0),
                         "gt_fg": int(gts[j].sum()), "pred_fg": int(p.sum())})
    macro = float(np.mean([float(r["dice"]) for r in rows])) if rows else float("nan")
    print(f"[{ts()}]   {setting} 评估完成：{len(rows)} 张，macro Dice={macro:.6f}，"
          f"用时 {time.time()-t0:.1f}s", flush=True)
    return rows


def load_source_model(src: str, device):
    """加载甲数据集 full 档位 checkpoint，校验 meta 一致后返回 (model, meta, ckpt_path)。"""
    ckpt = source_ckpt_path(src)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    n_full = full_pool_size(src)
    if meta.get("dataset") != src or meta.get("setting") != f"n={n_full}":
        raise SystemExit(f"[{ts()}] checkpoint meta 与 full 档位不符：{ckpt.name} -> {meta}")
    model = UNet().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != meta.get("n_params"):
        raise SystemExit(f"[{ts()}] 参数量不符：{n_params} != meta.n_params={meta.get('n_params')}")
    return model, meta, ckpt


def run_pair(src: str, tgt: str, device, force: bool = False) -> dict:
    """单个 (甲→乙) 组合：幂等推理 + marker 落盘。返回 marker 数据（或已有 marker）。"""
    setting = f"UNet({src}→{tgt})"
    expected = len(test_items(tgt))
    marker = MARKER_DIR / f"{src}_to_{tgt}.json"
    tag = f"[{setting}]"

    if not force and marker.exists() and count_rows(tgt, setting) == expected:
        data = json.loads(marker.read_text(encoding="utf-8"))
        print(f"[{ts()}] {tag} 已完成（marker + {expected} 行逐样本记录齐全，"
              f"macro Dice={data['test_macro_dice']:.6f}），跳过", flush=True)
        return data

    drop_rows(tgt, setting)
    print(f"[{ts()}] {tag} 开始：目标测试集 {expected} 张", flush=True)
    model, meta, ckpt = load_source_model(src, device)
    print(f"[{ts()}] {tag} 源模型 {ckpt.name}（n_train={meta['n_train']}，"
          f"best_epoch={meta['best_epoch']}，img_size={meta['img_size']}）", flush=True)
    rows = evaluate_cross(src, tgt, model, int(meta["img_size"]), device)
    if len(rows) != expected:
        raise SystemExit(f"[{ts()}] {tag} 行数不齐：{len(rows)} != {expected}")
    append_rows(rows)

    dices = [float(r["dice"]) for r in rows]
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "source": src, "target": tgt, "setting": setting,
        "source_checkpoint": str(ckpt.relative_to(ROOT)),
        "source_meta": {k: meta[k] for k in ("dataset", "setting", "n_train", "n_val",
                                             "best_epoch", "img_size", "batch_size",
                                             "lr", "seed", "precision", "n_params")
                        if k in meta},
        "n_test": expected,
        "test_macro_dice": round(float(np.mean(dices)), 6),
        "dice_min": round(float(np.min(dices)), 6),
        "dice_max": round(float(np.max(dices)), 6),
        "n_pred_empty": sum(1 for r in rows if int(r["pred_empty"]) == 1),
        "modality_gap": ("彩色内镜->灰度胸片（最大模态差）" if tgt == "Montgomery"
                         else "彩色内镜<->皮肤镜（近域跨集）"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    marker.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ts()}] {tag} 完成：macro Dice={data['test_macro_dice']:.6f}，"
          f"marker 已写 {marker.name}", flush=True)
    return data


# ---------------------------------------------------------------- 聚合
def read_metrics() -> list:
    if not METRICS_CSV.exists():
        return []
    with open(METRICS_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("experiment")]


def aggregate(pair_markers: dict) -> None:
    """重写 metrics.csv 的 cross_dataset 行 + summary.json 的 cross_dataset 键。"""
    rows = read_rows()

    # 1) U-Net 跨集行（从 per_sample_C.csv 独立聚合）
    unet_rows, unet_tree, incomplete = [], {}, []
    for src, tgt in PAIRS:
        setting = f"UNet({src}→{tgt})"
        expected = len(test_items(tgt))
        recs = [r for r in rows if r["dataset"] == tgt and r["setting"] == setting]
        if len(recs) != expected:
            incomplete.append(f"{setting}: {len(recs)}/{expected}")
            continue
        macro = float(np.mean([float(x["dice"]) for x in recs]))
        unet_rows.append({"experiment": "cross_dataset", "dataset": tgt, "setting": setting,
                          "n_samples": expected, "dice": f"{macro:.6f}"})
        m = pair_markers.get(setting, {})
        unet_tree[setting] = {
            "source": src, "target": tgt, "n_test": expected,
            "macro_dice": round(macro, 6),
            "n_pred_empty": sum(1 for x in recs if int(x["pred_empty"]) == 1),
        }
        for k in ("source_checkpoint", "source_meta", "dice_min", "dice_max", "modality_gap"):
            if k in m:
                unet_tree[setting][k] = m[k]
    if incomplete:
        raise SystemExit(f"[{ts()}] [ERROR] 以下组合样本数不齐，拒绝聚合: "
                         f"{'; '.join(incomplete)}（请先补跑）")

    # 2) MedSAM 零样本参考行：复用实验 A 同测试集（prompt_strategy/box）数字
    prompt_rows = [r for r in read_metrics()
                   if r["experiment"] == "prompt_strategy" and r["setting"] == "box"]
    medsam_src = {r["dataset"]: r for r in prompt_rows}
    medsam_rows, medsam_tree = [], {}
    for tgt in MEDSAM_TARGETS:
        expected = len(test_items(tgt))
        if tgt not in medsam_src:
            raise SystemExit(f"[{ts()}] [ERROR] metrics.csv 缺实验 A box 数字（{tgt}），"
                             f"无法复用 MedSAM 零样本参考，需先补测实验 A")
        r = medsam_src[tgt]
        if int(r["n_samples"]) != expected:
            raise SystemExit(f"[{ts()}] [ERROR] MedSAM 参考 n_samples 不一致（{tgt}: "
                             f"{r['n_samples']} != {expected}），不能作为同测试集对照")
        medsam_rows.append({"experiment": "cross_dataset", "dataset": tgt,
                            "setting": f"MedSAM({tgt})", "n_samples": int(r["n_samples"]),
                            "dice": r["dice"]})
        medsam_tree[tgt] = {"macro_dice": float(r["dice"]), "n_samples": int(r["n_samples"]),
                            "source": "results/metrics.csv experiment=prompt_strategy "
                                      "setting=box（实验 A 同测试集数字，直接复用）"}

    # 3) 同域 U-Net 参考数字（实验 B full 档位，来自 metrics.csv，便于论文对比衰减幅度）
    fewshot_full = {r["dataset"]: r for r in read_metrics()
                    if r["experiment"] == "fewshot"
                    and r["setting"] == f"n={full_pool_size(r['dataset'])}"}
    for setting, e in unet_tree.items():
        ref = fewshot_full.get(e["target"])
        if ref:
            e["unet_indomain_reference"] = {
                "macro_dice": float(ref["dice"]), "n_train": full_pool_size(e["target"]),
                "source": f"results/metrics.csv experiment=fewshot setting={ref['setting']}"}
            e["drop_vs_indomain"] = round(
                float(ref["dice"]) - e["macro_dice"], 6)

    # 4) metrics.csv：保留其他实验行，重写 cross_dataset 行
    keep = [r for r in read_metrics() if r["experiment"] != "cross_dataset"]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_COLS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in METRICS_COLS})
        for r in unet_rows + medsam_rows:
            w.writerow(r)
    print(f"[{ts()}] metrics.csv 已更新：cross_dataset {len(unet_rows)} UNet 行 + "
          f"{len(medsam_rows)} MedSAM 行（保留其他实验行 {len(keep)} 行）", flush=True)

    # 5) summary.json
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[{ts()}] [WARN] summary.json 读取失败（{e}），将重建", flush=True)
    summary["cross_dataset"] = {
        "experiment": "实验C：跨数据集泛化（U-Net 甲->乙 vs MedSAM 零样本乙）",
        "script": "scripts/cross_dataset.py",
        "date": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "pairs": [f"UNet({s} n={full_pool_size(s)}) -> {t} test {len(test_items(t))}"
                      for s, t in PAIRS],
            "rationale": "STATE.md Phase 1 建议：Kvasir-SEG<->ISIC2018 为彩色内镜 vs "
                         "皮肤镜近域跨集；->Montgomery 为彩色->灰度胸片最大模态差。"
                         "源模型取实验 B full 档位（n=全部训练池）checkpoint。",
            "modality_note": "Montgomery 为灰度胸片（PIL convert('RGB') 三通道复制），"
                             "测试集仅 28 张，结论需标注小样本",
        },
        "protocol": {
            "source_checkpoints": "checkpoints/unet_<ds>_<n>.pth（full 档位，含 meta；"
                                  "推理严格遵循 meta 的 img_size=256 / x/255 / mask>=128）",
            "inference": "与实验 B evaluate_test 逐行同协议：图像 PIL 双线性 256x256 "
                         "-> /255 -> UNet -> sigmoid -> 概率图双线性回原始分辨率 -> 0.5 "
                         "阈值；GT 为原始分辨率 >=128 二值化",
            "dice_convention": "与实验 A/B 一致：两空=1.0、恰一空=0.0、macro=算术平均",
            "test_subsets": "与实验 A/B 同一测试子集：Kvasir-SEG test 200 全量 / "
                            "ISIC2018 test 前 500/519 / Montgomery test 28 全量",
            "medsam_reuse": "MedSAM 零样本(乙) 数字复用实验 A prompt_strategy/box 同测试集"
                            "结果（脚本校验存在性与 n_samples 一致后写入；未做任何重推理）",
            "seed": SEED,
            "determinism": "纯推理无随机性；仍统一 seed=42；模型加载自实验 B 落盘权重",
        },
        "results": {"unet_cross": unet_tree, "medsam_zero_shot": medsam_tree},
        "per_sample_file": "results/per_sample_C.csv",
        "markers": "results/cross_runs/<src>_to_<tgt>.json",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"[{ts()}] summary.json 已更新（cross_dataset 键）", flush=True)

    # 6) 控制台汇总
    print(f"[{ts()}] ===== 实验 C 汇总（test macro Dice）=====", flush=True)
    for setting, e in unet_tree.items():
        ref = fewshot_full.get(e["target"])
        ref_s = (f"{float(ref['dice']):.6f}" if ref else "N/A（无同域U-Net）")
        ms = medsam_tree[e["target"]]["macro_dice"]
        print(f"{setting:<28s} = {e['macro_dice']:.6f}  | 同域U-Net(full)={ref_s}  "
              f"| MedSAM({e['target']})={ms:.6f}", flush=True)


# ---------------------------------------------------------------- 入口
def main() -> None:
    ap = argparse.ArgumentParser(description="实验C：跨数据集泛化")
    ap.add_argument("--probe", action="store_true",
                    help="试点：每组合前 4 张，不写任何结果文件")
    ap.add_argument("--force", action="store_true", help="强制重跑全部组合（忽略幂等）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    print(f"[{ts()}] 实验C 启动：python={'.'.join(map(str, sys.version_info[:3]))} "
          f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}",
          flush=True)
    print(f"[{ts()}] 组合：{['%s->%s' % p for p in PAIRS]}", flush=True)

    if args.probe:
        for src, tgt in PAIRS:
            model, meta, ckpt = load_source_model(src, device)
            print(f"[{ts()}] [PROBE] {ckpt.name} -> {tgt} test 前 4 张：", flush=True)
            rows = evaluate_cross(src, tgt, model, int(meta["img_size"]), device, limit=4)
            for rr in rows:
                print(f"[{ts()}]   [PROBE] {rr['sample_id']}: dice={rr['dice']} "
                      f"pred_fg={rr['pred_fg']} gt_fg={rr['gt_fg']}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[{ts()}] [PROBE] 结束（未写入任何结果文件）", flush=True)
        return

    pair_markers = {}
    for src, tgt in PAIRS:
        pair_markers[f"UNet({src}→{tgt})"] = run_pair(src, tgt, device, force=args.force)
    aggregate(pair_markers)
    print(f"[{ts()}] 实验C 本次调用结束", flush=True)


if __name__ == "__main__":
    main()
