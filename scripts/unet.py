# -*- coding: utf-8 -*-
"""实验B：U-Net 少样本学习曲线（实验方案 5.3 实验 B，Phase 3，论文图表数据来源）。

标准 2D U-Net（自实现）：4 层下采样，基础通道 64 → 瓶颈 1024，
DoubleConv = Conv3x3-BN-ReLU ×2，转置卷积上采样 + 跳连拼接，1x1 输出头。
训练：BCE+软Dice 损失，Adam lr=1e-4，batch 16，epoch<=200（实验方案原定 <=50，
经诊断该预算下从零训练严重欠收敛——train Dice 停在 0.41 且各学习率同平台——
统一放宽至 200，详见模块内 MAX_EPOCHS 注释与 STATE.md；早停 patience=5 仍生效），
早停 patience=5（监控**验证损失** BCE+软Dice：阈值化 val Dice 噪声过大曾致
随机提前终止；含 15 epoch 预热期；最佳权重按全程最低验证损失选取，
该点阈值化 val Dice 一并记录），seed=42。
性能说明：本机为功耗受限的笔记本 GPU（RTX 5060 Laptop，默认功耗墙 50W），
计算采用 bf16 混合精度（torch.autocast 包模型前向；损失/概率/Dice 全部 fp32），
这是纯性能优化，不改变训练协议，已记录于 summary.json 与 STATE.md。
OOM 兜底（实验方案 5.3/9）：显存探针不过 → batch 16→8 → 输入 256→224。
预处理：图像 PIL 双线性 resize 至 256x256 后 /255；mask 最近邻 resize + >=128
二值化（Phase 1 约定，与实验 A 完全一致）。轻度增强：训练期水平/垂直翻转各
p=0.5（图像与 mask 同步）；验证/测试不增强。

数据协议（与实验 A 完全一致，保证可比）：
  测试集：Kvasir-SEG test 200 全量；ISIC2018 test 取 splits.json 顺序
  （seed=42 洗牌产物）前 500/519。逐样本 Dice = 2|P∩G|/(|P|+|G|)，
  两空=1.0、恰一空=0.0，macro=算术平均。U-Net 概率图（256x256）双线性
  回原始分辨率后 0.5 阈值，与 MedSAM 推理协议对齐（公平比较）。

少样本档位（数据协议核心决策）：
  训练集（splits.json train，seed=42 洗牌顺序）末 1/10 固定为验证集
  （Kvasir-SEG 80 / ISIC2018 207，仅用于早停，任何档位都不参与训练），
  其余 90% 为训练池（720 / 1868）。
  档位 n=20 / 100 / full = 训练池前 n 张 / 全部（full 即 n=720 / n=1868）。

幂等：某档位 results/unet_runs/<ds>_<n>.json 标记 + checkpoints/*.pth +
  per_sample_B.csv 中该 (dataset, setting) 行数 == 测试集样本数 → 跳过训练；
  不齐则先删除该组合旧行，重训重测后重写。

用法：
  .venv/Scripts/python.exe scripts/unet.py --probe                    # 试点（不写任何结果）
  .venv/Scripts/python.exe scripts/unet.py --dataset Kvasir-SEG --tier 20
  .venv/Scripts/python.exe scripts/unet.py --dataset ISIC2018 --tier full
  .venv/Scripts/python.exe scripts/unet.py --aggregate-only           # 重算 metrics/summary
  .venv/Scripts/python.exe scripts/unet.py --figure-only              # 画学习曲线 pdf
（批量一键：bash scripts/run_experiment_B.sh）

实验C 说明：checkpoints/unet_<ds>_<n>.pth 内含 {"state_dict", "meta"}，
  meta 记录 img_size / 归一化（x/255）/ mask 二值化阈值等，跨数据集推理须遵循。
"""
import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
RUNS_DIR = RESULTS / "unet_runs"
PREDS_DIR = RESULTS / "preds_B"
PER_SAMPLE_CSV = RESULTS / "per_sample_B.csv"
METRICS_CSV = RESULTS / "metrics.csv"
SUMMARY_JSON = RESULTS / "summary.json"
FIG_DIR = ROOT / "figures"
FIG_PDF = FIG_DIR / "learning_curve.pdf"

SEED = 42
DATASETS = ["Kvasir-SEG", "ISIC2018"]
TIER_ORDER = ["20", "100", "full"]
MAX_TEST = 500                 # 与实验 A 一致：每数据集测试集最多 500 张
MAX_EPOCHS = 200
# epoch 上限说明：实验方案 5.2 原定 epoch<=50。实测（logs/unet_*_720.log 与 /tmp 诊断，
# 2026-10-24）：batch16 x 50 epoch 仅 2250 步（每图约 50 次参数更新），从零训练 31M
# U-Net 在 Kvasir 全量 720 张上 train Dice 停滞在 0.41（所有 lr 1e-4~1e-3 同一平台，
# 16 张过拟合测试证明模型/损失/数据全部健康），学习曲线近水平、"要多少标注追上
# MedSAM"无法回答。文献 Kvasir U-Net（Jha et al. 2020）batch2 x 100 epoch（每图
# 100+ 次更新）达 Dice 0.79。故统一将预算提至 200 epoch（早停 patience=5 仍生效），
# 其余超参严格按实验方案。此调整已记录于 STATE.md 与 results/summary.json。
PATIENCE = 5
EARLY_STOP_WARMUP = 15  # 早停预热期：验证 Dice 在欠训练期噪声大（实测逐 epoch 可在
                        # 0.0-0.5 间震荡），前 15 个 epoch 不计数 patience，防止严重
                        # 欠训练时被噪声性提前终止；最佳权重仍按全程最高验证 Dice 选取
LR = 1e-4
BASE_IMG_SIZE = 256
BASE_BATCH = 16
MIN_BATCH = 8                  # OOM 兜底第一级：batch 16→8
FALLBACK_IMG_SIZE = 224        # OOM 兜底第二级：256→224
VAL_DENOM = 10                 # 训练集末 1/10 固定为早停验证集
SAVE_PREDS_N = 20              # full 档位每数据集存 20 张预测 mask
MIN_FREE_MEM = 600 * 1024 * 1024  # 显存探针通过后要求的最低剩余（Adam 状态+碎片余量）
USE_AMP = True                 # bf16 混合精度（模型前向 bf16，损失/指标 fp32）


def autocast_ctx(device):
    """模型前向 autocast 上下文；损失与指标计算一律在 fp32（出上下文后 .float()）。"""
    if USE_AMP and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()

try:
    _RS = Image.Resampling
except AttributeError:  # Pillow < 9.1
    _RS = Image
BILINEAR = _RS.BILINEAR
NEAREST = _RS.NEAREST

_OOMErr = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
PER_SAMPLE_COLS = ["experiment", "dataset", "setting", "sample_id", "dice",
                   "pred_empty", "gt_fg", "pred_fg"]
METRICS_COLS = ["experiment", "dataset", "setting", "n_samples", "dice"]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_oom(err: BaseException) -> bool:
    return isinstance(err, _OOMErr) or "out of memory" in str(err).lower()


# ---------------------------------------------------------------- 数据
def load_splits(ds: str) -> dict:
    return json.loads((DATA / ds / "splits.json").read_text(encoding="utf-8"))


def train_items(ds: str) -> list:
    """训练集样本（splits.json train 顺序 = seed=42 洗牌产物）。"""
    sp = load_splits(ds)
    out = []
    for e in sp["train"]:
        if ds == "Kvasir-SEG":
            out.append({"sample_id": e, "img": DATA / ds / "images" / e,
                        "mask": DATA / ds / "masks" / e})
        else:  # ISIC2018
            out.append({"sample_id": e["image"], "img": DATA / ds / "images" / e["image"],
                        "mask": DATA / ds / "masks" / e["mask"]})
    return out


def test_items(ds: str) -> list:
    """测试集样本，与实验 A 完全同一协议（ISIC 取前 500/519）。"""
    sp = load_splits(ds)
    test = sp["test"]
    if ds == "Kvasir-SEG":
        return [{"sample_id": e, "img": DATA / ds / "images" / e,
                 "mask": DATA / ds / "masks" / e} for e in test]
    return [{"sample_id": e["image"], "img": DATA / ds / "images" / e["image"],
             "mask": DATA / ds / "masks" / e["mask"]} for e in test[:MAX_TEST]]


def resolve_tier(ds: str, tier: str) -> dict:
    """档位解析：返回 setting/n/训练子集/验证集等。"""
    items = train_items(ds)
    val_size = len(items) // VAL_DENOM
    pool = items[: len(items) - val_size]
    val = items[len(items) - val_size:]
    if tier == "full":
        sub, n = pool, len(pool)
    else:
        n = int(tier)
        if not (0 < n <= len(pool)):
            raise SystemExit(f"n={n} 超出训练池大小 {len(pool)}（{ds}）")
        sub = pool[:n]
    return {"setting": f"n={n}", "n": n, "sub": sub, "val": val,
            "total_train": len(items), "val_size": len(val), "pool_size": len(pool)}


def preload(items: list, img_size: int):
    """解码到内存：图像 (N,H,W,3) uint8（双线性），mask (N,H,W) uint8{0,1}（最近邻+>=128）。"""
    xs, ys = [], []
    t0 = time.time()
    for s in items:
        with Image.open(s["img"]) as im:
            im = im.convert("RGB").resize((img_size, img_size), BILINEAR)
            xs.append(torch.from_numpy(np.array(im, dtype=np.uint8)))
        with Image.open(s["mask"]) as m:
            m = m.convert("L").resize((img_size, img_size), NEAREST)
            ys.append((torch.from_numpy(np.array(m, dtype=np.uint8)) >= 128).to(torch.uint8))
    print(f"[{ts()}]   预加载 {len(items)} 对 -> {img_size}x{img_size}，用时 {time.time()-t0:.1f}s",
          flush=True)
    return torch.stack(xs), torch.stack(ys)


class TensorSegDataset(torch.utils.data.Dataset):
    """内存张量数据集；augment=True 时水平/垂直翻转各 p=0.5（图像与 mask 同步）。"""

    def __init__(self, xs, ys, augment: bool):
        self.xs, self.ys, self.augment = xs, ys, augment

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x, y = self.xs[i], self.ys[i]
        if self.augment:
            if random.random() < 0.5:  # 水平翻转（x: HWC 翻 W=dim2；y: HW 翻 W=dim1）
                x, y = torch.flip(x, [2]), torch.flip(y, [1])
            if random.random() < 0.5:  # 垂直翻转（x 翻 H=dim1；y 翻 H=dim0）
                x, y = torch.flip(x, [1]), torch.flip(y, [0])
        return x, y


# ---------------------------------------------------------------- 模型
class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin // 2 + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """标准 2D U-Net：4 层下采样，基础通道 64，瓶颈 1024，输出 1 通道 logit。"""

    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)
        self.enc2 = Down(b, 2 * b)
        self.enc3 = Down(2 * b, 4 * b)
        self.enc4 = Down(4 * b, 8 * b)
        self.bott = Down(8 * b, 16 * b)
        self.up4 = Up(16 * b, 8 * b, 8 * b)
        self.up3 = Up(8 * b, 4 * b, 4 * b)
        self.up2 = Up(4 * b, 2 * b, 2 * b)
        self.up1 = Up(2 * b, b, b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bt = self.bott(e4)
        d4 = self.up4(bt, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


# ---------------------------------------------------------------- 损失与指标
def soft_dice_loss(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


def dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """与实验 A 完全相同的 Dice 惯例：两空=1.0、恰一空=0.0。"""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = int(p.sum()), int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * int(np.logical_and(p, g).sum()) / (ps + gs)


@torch.no_grad()
def eval_val(model, xs, ys, device, batch):
    """验证集评估：返回 (val_loss, val_macro_dice)。
    val_loss = BCE+软Dice（无阈值，平滑单调，作早停/选模信号——阈值化 Dice 在欠训练期
    噪声极大（预测概率在 0.5 附近整体翻转），此前用它监控曾致 patience-5 随机触发）；
    val_macro_dice（256x256 阈值化）仅作记录参考。"""
    model.eval()
    tot_loss, n = 0.0, 0
    dices = []
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].permute(0, 3, 1, 2).to(device).float() / 255.0
        y = ys[i:i + batch].to(device).float()
        with autocast_ctx(device):
            lg = model(x).squeeze(1)
        lg = lg.float()
        loss = (F.binary_cross_entropy_with_logits(lg, y)
                + soft_dice_loss(lg.unsqueeze(1), y.unsqueeze(1)))
        tot_loss += float(loss.item()) * lg.shape[0]
        n += lg.shape[0]
        pred = torch.sigmoid(lg) > 0.5
        yb = y.bool()
        for j in range(pred.shape[0]):
            pj, gj = pred[j], yb[j]
            ps, gs = int(pj.sum()), int(gj.sum())
            if ps == 0 and gs == 0:
                dices.append(1.0)
            elif ps == 0 or gs == 0:
                dices.append(0.0)
            else:
                dices.append(2.0 * int((pj & gj).sum()) / (ps + gs))
    return tot_loss / n, (float(np.mean(dices)) if dices else float("nan"))


def memory_probe(model, batch: int, img_size: int, device) -> tuple:
    """零张量前向+反向显存探针（eval 模式，不污染 BN 统计、不更新权重）。
    返回 (ok, reason)。"""
    was = model.training
    model.eval()
    try:
        x = torch.zeros(batch, 3, img_size, img_size, device=device)
        y = torch.zeros(batch, img_size, img_size, device=device)
        with autocast_ctx(device):  # 与训练同精度，测真实峰值显存
            logits = model(x).squeeze(1).float()
        loss = (F.binary_cross_entropy_with_logits(logits, y)
                + soft_dice_loss(logits.unsqueeze(1), y.unsqueeze(1)))
        loss.backward()
        if device.type == "cuda":
            free = torch.cuda.mem_get_info()[0]
            if free <= MIN_FREE_MEM:
                return False, f"剩余显存仅 {free/2**30:.2f}GB (<{MIN_FREE_MEM/2**30:.1f}GB 余量线)"
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        if not is_oom(e):
            raise
        return False, f"CUDA OOM: {type(e).__name__}"
    finally:
        model.zero_grad(set_to_none=True)
        if was:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- 逐样本记录 IO
def read_rows() -> list:
    if not PER_SAMPLE_CSV.exists():
        return []
    with open(PER_SAMPLE_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("sample_id")]


def count_rows(ds: str, setting: str) -> int:
    return sum(1 for r in read_rows() if r["dataset"] == ds and r["setting"] == setting)


def drop_rows(ds: str, setting: str) -> None:
    """删除某 (dataset, setting) 的全部旧行（重训前清理，保证幂等且无陈旧数据）。"""
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


# ---------------------------------------------------------------- 测试评估
@torch.no_grad()
def evaluate_test(ds: str, setting: str, model, img_size: int, device,
                  save_pred_n: int = 0, limit: int = 0, batch: int = 32) -> list:
    """测试集逐样本评估：概率图双线性回原始分辨率 → 0.5 阈值 → 与原始分辨率 GT 求 Dice。
    与 MedSAM（实验 A）推理协议对齐。save_pred_n>0 时把前 n 张预测 mask 存 PNG。"""
    samples = test_items(ds)
    if limit:
        samples = samples[:limit]
    model.eval()
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
            rows.append({"experiment": "fewshot", "dataset": ds, "setting": setting,
                         "sample_id": s["sample_id"], "dice": f"{d:.6f}",
                         "pred_empty": int(p.sum() == 0),
                         "gt_fg": int(gts[j].sum()), "pred_fg": int(p.sum())})
            if save_pred_n and (i0 + j) < save_pred_n:
                outp = PREDS_DIR / ds / (Path(s["sample_id"]).stem + ".png")
                outp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(p.astype(np.uint8) * 255).save(outp)
    macro = float(np.mean([float(r["dice"]) for r in rows])) if rows else float("nan")
    print(f"[{ts()}]   测试评估完成：{len(rows)} 张，macro Dice={macro:.6f}，"
          f"用时 {time.time()-t0:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------- 训练（单档位）
def train_one(ds: str, tier: str, epochs: int, batch: int, img_size: int, probe: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    torch.backends.cudnn.benchmark = True

    r = resolve_tier(ds, tier)
    setting, n = r["setting"], r["n"]
    n_test_expected = len(test_items(ds))
    ckpt = CKPT_DIR / f"unet_{ds}_{n}.pth"
    marker = RUNS_DIR / f"{ds}_{n}.json"
    tag = f"[{ds} {setting}]"

    if probe:
        print(f"[{ts()}] [PROBE] {tag} {r['total_train']} 训练集 -> 验证 {r['val_size']}"
              f" + 训练池 {r['pool_size']}，本档位训练 {n}，测试集 {n_test_expected} 张", flush=True)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        if marker.exists() and ckpt.exists() and count_rows(ds, setting) == n_test_expected:
            print(f"[{ts()}] {tag} 已完成（marker+checkpoint+{n_test_expected} 行逐样本记录齐全），跳过",
                  flush=True)
            return
        drop_rows(ds, setting)
        print(f"[{ts()}] {tag} 开始：训练集 {r['total_train']}，验证集(末1/10，仅早停) {r['val_size']}，"
              f"训练池 {r['pool_size']}，本档位训练 {n}；epochs<={epochs} batch={batch} img={img_size}",
              flush=True)

    print(f"[{ts()}] {tag} 预加载训练/验证图像到内存...", flush=True)
    train_x, train_y = preload(r["sub"], img_size)
    val_x, val_y = preload(r["val"], img_size)

    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] {tag} U-Net 就绪：参数量 {n_params/1e6:.2f}M，device={device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 显存探针 + OOM 自动降级（batch 16→8 → 256→224，实验方案 5.3/9）
    oom_log = []
    while True:
        ok, reason = memory_probe(model, batch, img_size, device)
        if ok:
            break
        print(f"[{ts()}] {tag} [OOM兜底] 探针未通过：{reason}", flush=True)
        if batch > MIN_BATCH:
            batch //= 2
            oom_log.append(f"batch->{batch}")
            print(f"[{ts()}] {tag} [OOM兜底] batch 降为 {batch}", flush=True)
        elif img_size != FALLBACK_IMG_SIZE:
            img_size = FALLBACK_IMG_SIZE
            oom_log.append(f"img->{img_size}")
            print(f"[{ts()}] {tag} [OOM兜底] 分辨率降为 {img_size}，重新预加载", flush=True)
            train_x, train_y = preload(r["sub"], img_size)
            val_x, val_y = preload(r["val"], img_size)
        else:
            raise RuntimeError(f"{tag} batch={batch} img={img_size} 仍 OOM，需人工介入")
    print(f"[{ts()}] {tag} 显存探针通过：batch={batch} img={img_size}"
          f"{'（降级: ' + ','.join(oom_log) + '）' if oom_log else ''}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    loader = torch.utils.data.DataLoader(
        TensorSegDataset(train_x, train_y, augment=True),
        batch_size=batch, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(SEED))

    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    best_dice_at_best = float("nan")
    t0 = time.time()
    ep = 0
    for ep in range(1, epochs + 1):
        model.train()
        loss_tot, nb, inter_t, p_t, g_t = 0.0, 0, 0, 0, 0
        for xb, yb in loader:
            x = xb.permute(0, 3, 1, 2).to(device, non_blocking=True).float() / 255.0
            y = yb.to(device, non_blocking=True).float()
            with autocast_ctx(device):
                logits = model(x).squeeze(1)
            logits = logits.float()  # 损失与指标保持 fp32
            loss = (F.binary_cross_entropy_with_logits(logits, y)
                    + soft_dice_loss(logits.unsqueeze(1), y.unsqueeze(1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = torch.sigmoid(logits) > 0.5
                yb_bool = y.bool()
                inter_t += int((pred & yb_bool).sum())
                p_t += int(pred.sum())
                g_t += int(yb_bool.sum())
            loss_tot += float(loss.item())
            nb += 1
        train_dice = 2.0 * inter_t / (p_t + g_t + 1e-8)
        val_loss, val_dice = eval_val(model, val_x, val_y, device, max(batch, 16))
        if val_loss < best_loss:  # 早停/选模信号：验证损失（平滑）；Dice 仅记录
            best_loss, best_epoch, bad = val_loss, ep, 0
            best_dice_at_best = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif ep >= EARLY_STOP_WARMUP:  # 预热期内不计数（防噪声导致欠训练提前终止）
            bad += 1
        mem = (torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == "cuda" else 0.0
        print(f"[{ts()}] {tag} epoch {ep:02d}/{epochs} loss={loss_tot/max(nb,1):.4f} "
              f"train_dice(micro)={train_dice:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
              f"best(loss)={best_loss:.4f}@{best_epoch}(dice={best_dice_at_best:.4f}) bad={bad} "
              f"mem={mem:.2f}GB elapsed={time.time()-t0:.0f}s", flush=True)
        if bad >= PATIENCE and ep >= EARLY_STOP_WARMUP:
            print(f"[{ts()}] {tag} 早停：预热 {EARLY_STOP_WARMUP} epoch 后连续 {PATIENCE} "
                  f"个 epoch 验证损失无下降", flush=True)
            break

    assert best_state is not None, "至少应记录一个 epoch"
    model.load_state_dict(best_state)
    print(f"[{ts()}] {tag} 训练结束：best val_loss={best_loss:.6f} @ epoch {best_epoch}"
          f"（该点 val_dice={best_dice_at_best:.4f}；共跑 {ep} epoch，用时 {time.time()-t0:.0f}s）", flush=True)

    if probe:
        rows = evaluate_test(ds, setting, model, img_size, device, save_pred_n=0, limit=8)
        for rr in rows:
            print(f"[{ts()}]   [PROBE] {rr['sample_id']}: dice={rr['dice']} "
                  f"pred_fg={rr['pred_fg']} gt_fg={rr['gt_fg']}", flush=True)
        if device.type == "cuda":
            print(f"[{ts()}] [PROBE] 峰值显存（训练全程）: "
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f}GB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f}GB", flush=True)
        print(f"[{ts()}] [PROBE] 结束（未写入任何结果文件）", flush=True)
        return

    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "meta": {
        "dataset": ds, "setting": setting, "n_train": n, "n_val": r["val_size"],
        "pool_size": r["pool_size"], "total_train": r["total_train"],
        "best_epoch": best_epoch, "best_val_loss": round(best_loss, 6),
        "val_dice_at_best": round(best_dice_at_best, 6),
        "batch_size": batch, "img_size": img_size, "lr": LR, "seed": SEED,
        "precision": "bf16 autocast（模型前向 bf16，损失/指标 fp32）" if USE_AMP else "fp32",
        "normalization": "x/255（PIL 双线性 resize 至 img_size）",
        "mask_rule": "convert('L') 最近邻 resize + >=128 二值化",
        "n_params": n_params,
    }}, ckpt)
    print(f"[{ts()}] {tag} 最佳模型已存 {ckpt.name}", flush=True)

    rows = evaluate_test(ds, setting, model, img_size, device,
                         save_pred_n=(SAVE_PREDS_N if tier == "full" else 0))
    append_rows(rows)

    marker_data = {
        "dataset": ds, "setting": setting, "n_train": n, "n_val": r["val_size"],
        "pool_size": r["pool_size"], "total_train": r["total_train"],
        "best_epoch": best_epoch, "epochs_run": ep,
        "best_val_loss": round(best_loss, 6),
        "val_dice_at_best": round(best_dice_at_best, 6),
        "batch_size": batch, "img_size": img_size, "lr": LR, "max_epochs": epochs,
        "patience": PATIENCE, "early_stop_warmup": EARLY_STOP_WARMUP, "seed": SEED,
        "early_stop_monitor": "val_loss（BCE+软Dice，无阈值，平滑）",
        "n_params": n_params,
        "precision": "bf16 autocast（模型前向 bf16，损失/指标 fp32）" if USE_AMP else "fp32",
        "oom_downgrades": oom_log,
        "test_macro_dice": round(float(np.mean([float(x["dice"]) for x in rows])), 6),
        "n_test": len(rows),
        "n_pred_empty": sum(1 for x in rows if int(x["pred_empty"]) == 1),
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "augmentation": "h/v flip p=0.5（仅训练期）",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    marker.write_text(json.dumps(marker_data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ts()}] {tag} 完成：test macro Dice={marker_data['test_macro_dice']:.6f}，"
          f"marker 已写 {marker.name}", flush=True)


# ---------------------------------------------------------------- 聚合
def aggregate() -> None:
    rows = read_rows()
    by: dict = {}
    for r in rows:
        by.setdefault((r["dataset"], r["setting"]), []).append(r)

    metrics_rows, tree, incomplete = [], {}, []
    for ds in DATASETS:
        expected = len(test_items(ds))
        tree[ds] = {}
        for tier in TIER_ORDER:
            rr = resolve_tier(ds, tier)
            setting, n = rr["setting"], rr["n"]
            recs = by.get((ds, setting), [])
            if len(recs) != expected:
                incomplete.append(f"{ds}x{setting}: {len(recs)}/{expected}")
                continue
            macro = float(np.mean([float(x["dice"]) for x in recs]))
            n_empty = sum(1 for x in recs if int(x["pred_empty"]) == 1)
            entry = {"n_train": n, "n_test": expected, "macro_dice": round(macro, 6),
                     "n_pred_empty": n_empty}
            marker = RUNS_DIR / f"{ds}_{n}.json"
            if marker.exists():
                meta = json.loads(marker.read_text(encoding="utf-8"))
                for k in ("best_epoch", "epochs_run", "best_val_loss", "val_dice_at_best",
                          "batch_size", "img_size", "n_val", "pool_size", "total_train",
                          "checkpoint", "precision", "oom_downgrades", "early_stop_monitor"):
                    if k in meta:
                        entry[k] = meta[k]
            tree[ds][setting] = entry
            metrics_rows.append({"experiment": "fewshot", "dataset": ds, "setting": setting,
                                 "n_samples": expected, "dice": f"{macro:.6f}"})
    if incomplete:
        print(f"[{ts()}] [WARN] 以下组合样本数不齐，本轮不写其 metrics（补跑后自动补齐）: "
              f"{'; '.join(incomplete)}", flush=True)

    # metrics.csv：保留其他实验的行，重写 fewshot 的行
    keep = []
    if METRICS_CSV.exists():
        with open(METRICS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") != "fewshot":
                    keep.append(row)
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_COLS)
        w.writeheader()
        for row in keep:
            w.writerow({k: row.get(k, "") for k in METRICS_COLS})
        for row in metrics_rows:
            w.writerow(row)
    print(f"[{ts()}] metrics.csv 已更新：fewshot {len(metrics_rows)} 行"
          f"（保留其他实验行 {len(keep)} 行）", flush=True)

    # MedSAM 零参考线数字（取自 metrics.csv，与实验 A 同源）
    medsam_ref = {}
    if METRICS_CSV.exists():
        with open(METRICS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("experiment") == "prompt_strategy" and row.get("setting") == "box":
                    medsam_ref[row["dataset"]] = {
                        "macro_dice": float(row["dice"]),
                        "n_samples": int(row["n_samples"]),
                        "source": "results/metrics.csv (experiment=prompt_strategy, setting=box)"}

    # summary.json
    summary = {}
    if SUMMARY_JSON.exists():
        try:
            summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[{ts()}] [WARN] summary.json 读取失败（{e}），将重建", flush=True)
    n_params_m = None
    mk = RUNS_DIR / "Kvasir-SEG_20.json"
    if mk.exists():
        n_params_m = json.loads(mk.read_text(encoding="utf-8")).get("n_params")
    summary["fewshot"] = {
        "experiment": "实验B：U-Net 少样本学习曲线",
        "script": "scripts/unet.py",
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "name": "标准 2D U-Net（自实现，scripts/unet.py）",
            "architecture": "4 层下采样，基础通道 64 -> 瓶颈 1024，"
                            "DoubleConv=Conv3x3-BN-ReLU x2，转置卷积上采样+跳连，1x1 输出头",
            "n_params": n_params_m,
            "loss": "BCEWithLogits + 软 Dice（L = BCE + 1 - softDice，eps=1）",
            "optimizer": f"Adam lr={LR}",
            "batch_size": "16（显存探针不过自动降 8）",
            "input": "256x256（OOM 兜底降 224）",
            "max_epochs": MAX_EPOCHS,
            "epoch_budget_note": "实验方案原定 epoch<=50；实测该预算下 U-Net 严重欠收敛"
                                  "（Kvasir 全量 train Dice 停于 0.41，各 lr 同平台；16 张过拟合测试证明"
                                  "模型/损失/数据健康；根因为每图仅约 50 次更新，文献 Kvasir U-Net 每图 "
                                  "100+ 次更新达 Dice 0.79），为保证少样本学习曲线可回答论文核心问题，"
                                  "统一放宽至 200 epoch，其余超参不变（诊断记录见 logs/unet_*.log 与 STATE.md）",
            "early_stopping": f"patience={PATIENCE}，预热 {EARLY_STOP_WARMUP} epoch，监控验证损失"
                              "（BCE+软Dice，无阈值，平滑；阈值化 val Dice 在欠训练期噪声极大——预测概率在 "
                              "0.5 附近整体翻转，曾致 patience-5 在 ~ep20 随机触发、200 epoch 预算用不满），"
                              "mode=min，恢复最佳权重；阈值化 val Dice 同时记录（marker 的 val_dice_at_best）",
        },
        "protocol": {
            "splits": "splits.json 训练集（seed=42 洗牌顺序）末 1/10 固定为验证集（仅早停，"
                      "任何档位不参与训练）：Kvasir-SEG 80 / ISIC2018 207；其余 90% 为训练池：720 / 1868",
            "tiers": "n=20 / 100 = 训练池前 n 张；full = 训练池全部（Kvasir-SEG n=720 / ISIC2018 n=1868）",
            "preprocess": "图像 PIL 双线性 resize 至 256x256 后 /255；mask 最近邻 resize + >=128 二值化"
                          "（Phase 1 约定，与实验 A 一致）",
            "augmentation": "训练期水平/垂直翻转各 p=0.5（图像与 mask 同步）；验证/测试不增强",
            "test_protocol": "与实验 A 同一测试集（Kvasir-SEG test 200 全量；ISIC2018 test 前 500/519）"
                             "与同一 Dice 惯例（两空=1.0、恰一空=0.0、macro=算术平均）；"
                             "U-Net 概率图（256x256）双线性回原始分辨率后 0.5 阈值，与 MedSAM 推理协议对齐",
            "seed": SEED,
            "determinism": "random/numpy/torch/cuda 全部 seed=42；DataLoader shuffle 用独立 "
                           "torch.Generator().manual_seed(42)；cudnn.benchmark=True（速度优先）",
            "precision": "bf16 混合精度（torch.autocast 包模型前向；损失/概率/Dice 计算 fp32）。"
                         "原因：本机 RTX 5060 Laptop 功耗受限（默认功耗墙 50W，实测满载 30W/1080MHz，"
                         "fp32 单步 1.0-1.7s、bf16 0.43s），属性能优化，不改变训练协议",
            "power_note": "训练期已将 Windows 电源计划由'平衡'切至'高性能'（GPU 功耗 30W→满载提升，"
                          "实测 fp32 提速约 45%），复现实验时建议同样设置",
            "oom_fallback": "启动显存探针（零张量前向+反向，不污染权重/BN）不过则 batch 16->8、"
                            "再 256->224；各档位实际配置见 results/unet_runs/*.json 的 batch_size/img_size",
        },
        "results": tree,
        "medsam_box_reference": medsam_ref,
        "checkpoints": "checkpoints/unet_<dataset>_<n>.pth（含 state_dict + meta，实验C 跨数据集"
                       "推理须遵循 meta 内 img_size/归一化/mask 规则）",
        "per_sample_file": "results/per_sample_B.csv",
        "pred_masks": f"full 档位模型在每数据集前 {SAVE_PREDS_N} 张测试样本的预测 mask："
                      "results/preds_B/<dataset>/<sample_stem>.png（0/255，原始分辨率，与 preds_A 同样本便于四列对比）",
        "figure": "figures/learning_curve.pdf（+png，300dpi，由 --figure-only 从 metrics.csv 生成）",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ts()}] summary.json 已更新（fewshot 键）", flush=True)

    # 控制台汇总表
    print(f"[{ts()}] ===== 实验 B 汇总（test macro Dice）=====", flush=True)
    for ds in DATASETS:
        cells = []
        for tier in TIER_ORDER:
            setting = resolve_tier(ds, tier)["setting"]
            e = tree.get(ds, {}).get(setting)
            cells.append(f"{setting}={e['macro_dice']:.6f}" if e else f"{setting}=N/A")
        ref = medsam_ref.get(ds, {}).get("macro_dice")
        ref_s = f"{ref:.6f}" if ref is not None else "N/A"
        print(f"{ds:<12s} " + "  ".join(cells) + f"  | MedSAM(box)={ref_s}", flush=True)


# ---------------------------------------------------------------- 学习曲线图
def make_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    with open(METRICS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fewshot, n_test_map = {}, {}
    for r in rows:
        if r.get("experiment") == "fewshot":
            fewshot[(r["dataset"], r["setting"])] = float(r["dice"])
            n_test_map[r["dataset"]] = int(r["n_samples"])
    medsam = {r["dataset"]: float(r["dice"]) for r in rows
              if r.get("experiment") == "prompt_strategy" and r.get("setting") == "box"}

    missing = [f"{ds}x{resolve_tier(ds, t)['setting']}"
               for ds in DATASETS for t in TIER_ORDER
               if (ds, resolve_tier(ds, t)["setting"]) not in fewshot]
    if missing:
        raise SystemExit(f"[{ts()}] 缺少 fewshot 结果，无法画图: {'; '.join(missing)}")
    if set(DATASETS) - set(medsam):
        raise SystemExit(f"[{ts()}] metrics.csv 缺少 prompt_strategy/box 参考线数字: {DATASETS}")

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ds in zip(axes, DATASETS):
        pts = sorted((int(s[2:]), d) for (dds, s), d in fewshot.items() if dds == ds)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=6, zorder=3,
                label="U-Net（从零训练，BCE+Dice，早停）")
        for x, y in pts:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color="#1f77b4")
        ref = medsam[ds]
        ax.axhline(ref, color="#d62728", ls="--", lw=1.5, zorder=2,
                   label=f"MedSAM 零样本（GT 框提示）= {ref:.4f}")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] / 1.6, xs[-1] * 1.8)
        ax.set_ylim(0.0, 1.02)  # 两子图统一 y 轴范围，便于横向对比
        ax.set_xlabel("训练标注样本数 n（对数刻度）", fontsize=11)
        ax.set_ylabel("macro Dice（测试集）", fontsize=11)
        ax.set_title(f"{ds}（测试 {n_test_map.get(ds, '?')} 张）", fontsize=12)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_PDF, dpi=300)
    fig.savefig(FIG_PDF.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[{ts()}] 学习曲线已存 {FIG_PDF}（+png，300dpi）", flush=True)


# ---------------------------------------------------------------- 入口
def main() -> None:
    ap = argparse.ArgumentParser(description="实验B：U-Net 少样本学习曲线")
    ap.add_argument("--dataset", choices=DATASETS, help="数据集")
    ap.add_argument("--tier", choices=TIER_ORDER, help="档位：20 / 100 / full")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH)
    ap.add_argument("--img-size", type=int, default=BASE_IMG_SIZE)
    ap.add_argument("--probe", action="store_true", help="试点：2 epoch + 8 张测试，不写任何结果")
    ap.add_argument("--aggregate-only", action="store_true", help="仅重算 metrics/summary")
    ap.add_argument("--figure-only", action="store_true", help="仅画学习曲线")
    args = ap.parse_args()

    print(f"[{ts()}] 实验B 启动：python={'.'.join(map(str, sys.version_info[:3]))} "
          f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if args.probe:
        train_one("Kvasir-SEG", "20", epochs=2, batch=args.batch_size,
                  img_size=args.img_size, probe=True)
    elif args.aggregate_only:
        aggregate()
    elif args.figure_only:
        make_figure()
    else:
        if not args.dataset or not args.tier:
            ap.error("需要 --dataset + --tier，或 --probe / --aggregate-only / --figure-only")
        train_one(args.dataset, args.tier, epochs=args.epochs,
                  batch=args.batch_size, img_size=args.img_size)
        aggregate()
    print(f"[{ts()}] 实验B 本次调用结束", flush=True)


if __name__ == "__main__":
    main()
