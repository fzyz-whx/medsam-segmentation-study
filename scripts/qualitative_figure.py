# -*- coding: utf-8 -*-
"""Phase 4：定性可视化 figures/qualitative.pdf（论文图 3）。

布局：三个数据集块（Kvasir-SEG、ISIC2018、Montgomery），每块 2 好 + 2 差 case
× 四列（原图 / GT / MedSAM 框提示 / U-Net 全量档）。上行两块并排（Kvasir|ISIC），
下行 Montgomery 块 + 右侧图例/说明面板。

素材（全部为 results/ 与 data/ 下落盘文件）：
  - 预测 mask：results/preds_A/<ds>/<stem>.png（MedSAM box，实验A存的前 20 张测试样本）、
               results/preds_B/<ds>/<stem>.png（U-Net full 档，实验B存的前 20 张，与 preds_A 同样本）
  - 逐样本 Dice（面板角标与选例依据）：results/per_sample_A.csv（setting=box）、
               results/per_sample_B.csv（setting=n=720 / n=1868）
  - 原图与 GT：data/<ds>/images|masks/（ISIC mask 名查 splits.json）

选例协议（如实记录于 results/qualitative_cases.json）：
  仅在已保存预测的前 20 张测试样本内挑选；Kvasir-SEG/ISIC2018 按
  (MedSAM box Dice + U-Net full Dice)/2 排序——好 case=最高 2 例，差 case=最低 2 例
  （并列按 sample_id 升序）；Montgomery 无 U-Net 预测，按 MedSAM box Dice 排序。
  Montgomery 的 U-Net 列如实留白注记（U-Net 未在该数据集训练，无全量档模型）。

显示约定：所有图 letterbox 到正方形白底（保持纵横比）；GT=红色半透明覆盖+实色边界，
MedSAM=绿色，U-Net=蓝色；每预测面板左下角标注该样本逐样本 Dice（取自 per_sample CSV，
不重算）。SimHei 中文，300dpi，PDF+PNG 双落盘。

幂等：全量重算重写（素材为只读 CSV/PNG，输出确定）。
用法：.venv/Scripts/python.exe scripts/qualitative_figure.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREDS_A = RESULTS / "preds_A"
PREDS_B = RESULTS / "preds_B"
FIG_DIR = ROOT / "figures"
FIG_PDF = FIG_DIR / "qualitative.pdf"
CASES_JSON = RESULTS / "qualitative_cases.json"

RENDER = 512          # letterbox 目标边长（像素）
CELL_IN = 1.45        # 每面板边长（英寸）
GT_COLOR, MEDSAM_COLOR, UNET_COLOR = (220, 40, 40), (30, 170, 60), (40, 90, 220)
ALPHA = 0.42

DS_INFO = {
    "Kvasir-SEG": "Kvasir-SEG（彩色内镜息肉，测试 200 张）",
    "ISIC2018": "ISIC2018（皮肤镜病变，测试 500/519 张）",
    "Montgomery": "Montgomery（灰度胸片肺野，测试 28 张）",
}
COL_TITLES = ["原图", "GT", "MedSAM（框提示）", "U-Net（全量档）"]


def dice_map(csv_name: str, ds: str, setting: str) -> dict:
    """{stem: (sample_id, dice)}，取自 results/ 逐样本 CSV（权威数字来源）。"""
    out = {}
    with open(RESULTS / csv_name, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == ds and r["setting"] == setting:
                stem = Path(r["sample_id"]).stem
                out[stem] = (r["sample_id"], float(r["dice"]))
    return out


def pick_cases(ds: str) -> list:
    """返回 [{stem, sample_id, kind, dice_medsam, dice_unet(or None), score}]，好在前。"""
    a = dice_map("per_sample_A.csv", ds, "box")
    stems = {p.stem for p in (PREDS_A / ds).glob("*.png")}
    if ds in ("Kvasir-SEG", "ISIC2018"):
        setting_b = "n=720" if ds == "Kvasir-SEG" else "n=1868"
        b = dice_map("per_sample_B.csv", ds, setting_b)
        stems_b = {p.stem for p in (PREDS_B / ds).glob("*.png")}
        if stems_b != stems:
            raise SystemExit(f"{ds}: preds_A 与 preds_B 样本不一致 {len(stems)}/{len(stems_b)}")
        common = sorted(stems & set(a) & set(b))
        if len(common) < len(stems):
            print(f"[qualitative] [WARN] {ds}: {len(stems)-len(common)} 张缺逐样本记录，剔除")
        score = {s: (a[s][1] + b[s][1]) / 2.0 for s in common}
    else:  # Montgomery：无 U-Net，按 MedSAM box Dice 排序
        common = sorted(stems & set(a))
        score = {s: a[s][1] for s in common}
    ranked = sorted(common, key=lambda s: (-score[s], s))
    good, bad = ranked[:2], ranked[-2:][::-1]  # 差① = 最差
    out = []
    for kind, group in (("好", good), ("差", bad)):
        for i, s in enumerate(group, 1):
            out.append({"kind": f"{kind}{'①②'[i-1]}", "stem": s,
                        "sample_id": a[s][0], "score": round(score[s], 6),
                        "dice_medsam": round(a[s][1], 6),
                        "dice_unet": round(b[s][1], 6) if ds in ("Kvasir-SEG", "ISIC2018") else None})
    print(f"[qualitative] {ds}: " + "; ".join(
        f"{c['kind']} {c['stem'][:20]} score={c['score']:.4f} "
        f"(MedSAM={c['dice_medsam']:.4f}, U-Net={'-' if c['dice_unet'] is None else format(c['dice_unet'], '.4f')})"
        for c in out), flush=True)
    return out


def mask_path(ds: str, sample_id: str) -> Path:
    if ds == "ISIC2018":  # mask 文件名查 splits.json（{"image","mask"} 对）
        sp = json.loads((DATA / ds / "splits.json").read_text(encoding="utf-8"))
        for e in sp["test"]:
            if e["image"] == sample_id:
                return DATA / ds / "masks" / e["mask"]
        raise SystemExit(f"ISIC splits.json 找不到 {sample_id}")
    return DATA / ds / "masks" / sample_id


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_gt(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        return np.array(m.convert("L")) >= 128  # Phase 1 约定


def load_pred(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        return np.array(m.convert("L")) >= 128  # preds 存为 0/255


def boundary(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return m & ~e


def render(img: np.ndarray, mask: np.ndarray | None, color, size: int = RENDER) -> np.ndarray:
    """letterbox 到 size×size 白底；mask 作半透明覆盖 + 实色边界。"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    base = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR)).copy()
    if mask is not None:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                       .resize((nw, nh), Image.NEAREST)) >= 128
        fill = np.zeros((nh, nw, 3), np.float32)
        a = np.zeros((nh, nw, 1), np.float32)
        fill[m] = color
        a[m] = ALPHA
        base = (base * (1 - a) + fill * a).astype(np.uint8)
        bd = boundary(m)
        base[bd] = color  # 边界实色
    canvas = np.full((size, size, 3), 255, np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = base
    return canvas


def draw_panel(ax, img, mask, color, dice=None):
    ax.imshow(render(img, mask, color), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#888888"), sp.set_linewidth(0.6)
    if dice is not None:
        ax.text(0.03, 0.03, f"Dice={dice:.4f}", transform=ax.transAxes,
                fontsize=7.5, va="bottom", ha="left", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#999999",
                          lw=0.5, alpha=0.9))


def main() -> None:
    # ---------- 选例（含校验） ----------
    cases = {ds: pick_cases(ds) for ds in DS_INFO}
    CASES_JSON.write_text(json.dumps({
        "selection_protocol": (
            "仅在 results/preds_A、preds_B 已保存预测的前 20 张测试样本内挑选；"
            "Kvasir-SEG/ISIC2018 按 (MedSAM box Dice + U-Net full Dice)/2 排序，"
            "好 case=最高 2 例、差 case=最低 2 例（并列按 sample_id 升序）；"
            "Montgomery 无 U-Net 预测，按 MedSAM box Dice 排序。"
            "Dice 数值取自 results/per_sample_A.csv（setting=box）与 "
            "results/per_sample_B.csv（setting=n=720/n=1868），未重算。"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "cases": {ds: cs for ds, cs in cases.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[qualitative] 选例清单已写 {CASES_JSON}", flush=True)

    # ---------- 布局常量（英寸，反推 figsize） ----------
    # GAP_IN 须容纳右块行标签（右对齐贴左缘），LEFT_IN 须容纳左块行标签
    LEFT_IN, GAP_IN, MARG_IN = 1.00, 1.05, 0.28
    TITLE_IN, HEADER_IN, BLOCK_GAP_IN = 0.30, 0.26, 0.42
    block_w = 4 * CELL_IN
    fig_w = MARG_IN + LEFT_IN + block_w + GAP_IN + block_w + MARG_IN
    fig_h = 0.34 + (TITLE_IN + HEADER_IN + 4 * CELL_IN) * 2 + BLOCK_GAP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    print(f"[qualitative] figsize={fig_w:.2f} x {fig_h:.2f} in", flush=True)

    def xfrac(inch):  # 英寸 → 图宽比例
        return inch / fig_w

    def yfrac_from_top(inch):  # 距顶部英寸 → matplotlib bottom 比例
        return 1.0 - inch / fig_h

    blocks = [("Kvasir-SEG", MARG_IN + LEFT_IN, 0.0),
              ("ISIC2018", MARG_IN + LEFT_IN + block_w + GAP_IN, 0.0),
              ("Montgomery", MARG_IN + LEFT_IN, 1.0)]
    top_in = 0.30  # suptitle 高度
    for ds, x0_in, row in blocks:
        y_top_in = top_in + row * (TITLE_IN + HEADER_IN + 4 * CELL_IN + BLOCK_GAP_IN)
        # 块标题
        fig.text(xfrac(x0_in + block_w / 2), yfrac_from_top(y_top_in + 0.20),
                 DS_INFO[ds], ha="center", va="center", fontsize=12, fontweight="bold")
        # 列头
        for ci, t in enumerate(COL_TITLES):
            fig.text(xfrac(x0_in + ci * CELL_IN + CELL_IN / 2),
                     yfrac_from_top(y_top_in + TITLE_IN + HEADER_IN / 2),
                     t, ha="center", va="center", fontsize=10)
        # 行
        for ri, c in enumerate(cases[ds]):
            y0_in = y_top_in + TITLE_IN + HEADER_IN + ri * CELL_IN
            sid, stem = c["sample_id"], c["stem"]
            img = load_rgb(DATA / ds / "images" / sid)
            gt = load_gt(mask_path(ds, sid))
            pa = PREDS_A / ds / f"{stem}.png"
            pb = PREDS_B / ds / f"{stem}.png"
            # 行标签（块左侧，右对齐贴块左缘，不与相邻块重叠）
            fig.text(xfrac(x0_in - 0.08), yfrac_from_top(y0_in + CELL_IN / 2),
                     f"{c['kind']}\n{stem[:14]}…" if len(stem) > 14 else f"{c['kind']}\n{stem}",
                     ha="right", va="center", fontsize=8.5)
            for ci in range(4):
                ax = fig.add_axes([xfrac(x0_in + ci * CELL_IN),
                                   yfrac_from_top(y0_in + CELL_IN),
                                   CELL_IN / fig_w, CELL_IN / fig_h])
                if ci == 0:
                    draw_panel(ax, img, None, None)
                elif ci == 1:
                    draw_panel(ax, img, gt, GT_COLOR)
                elif ci == 2:
                    if pa.exists():
                        draw_panel(ax, img, load_pred(pa), MEDSAM_COLOR, c["dice_medsam"])
                    else:  # 缺预测如实留白注记
                        blank_note(ax, "无预测\n（preds_A 缺失）")
                else:  # U-Net 列
                    if pb.exists():
                        draw_panel(ax, img, load_pred(pb), UNET_COLOR, c["dice_unet"])
                    else:  # Montgomery：U-Net 未在该数据集训练
                        blank_note(ax, "无预测\n（U-Net 未在该\n数据集训练，\n无全量档模型）")

    # ---------- 右下说明面板 ----------
    lx_in = MARG_IN + LEFT_IN + block_w + GAP_IN
    ly_in = top_in + (TITLE_IN + HEADER_IN + 4 * CELL_IN) + BLOCK_GAP_IN
    ax = fig.add_axes([xfrac(lx_in), yfrac_from_top(ly_in + 4 * CELL_IN),
                       block_w / fig_w, 4 * CELL_IN / fig_h])
    ax.set_facecolor("#f7f7f7")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#aaaaaa"), sp.set_linewidth(0.8)
    # 注：多行 text 的 bbox 宽度在部分后端不可靠，故用 axes 背景与边框代替 bbox
    notes = (
        "图例与说明\n"
        "──────────────────────────────\n"
        "覆盖颜色：红 = GT；绿 = MedSAM（框提示）\n"
        "预测；蓝 = U-Net（全量档）预测\n"
        "（半透明填充 + 实色边界）。\n"
        "面板左下角为该样本逐样本 Dice，\n"
        "取自 results/per_sample_A.csv（box）\n"
        "与 results/per_sample_B.csv\n"
        "（n=720 / n=1868），未重算。\n"
        "──────────────────────────────\n"
        "好/差选例协议：仅在已保存预测的\n"
        "前 20 张测试样本内，按 (MedSAM +\n"
        "U-Net) 逐样本 Dice 均值排序\n"
        "（Montgomery 仅 MedSAM）：\n"
        "好 = 最高 2 例，差 = 最低 2 例。\n"
        "完整选例清单：\n"
        "results/qualitative_cases.json。\n"
        "──────────────────────────────\n"
        "Montgomery 的 U-Net 列留白：\n"
        "U-Net 未在该数据集上训练\n"
        "（无全量档模型），仅参与实验 C\n"
        "跨数据集推理。\n"
        f"生成：scripts/qualitative_figure.py\n"
        f"（{datetime.now().date()}，SimHei，300dpi）")
    ax.text(0.03, 0.985, notes, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, linespacing=1.5)

    fig.suptitle("定性分割对比：MedSAM 零样本（框提示） vs U-Net（全量标注档）", fontsize=14,
                 y=yfrac_from_top(0.12))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_PDF, dpi=300)
    fig.savefig(FIG_PDF.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[qualitative] 已写 {FIG_PDF}（+png，{RENDER}px/面板，300dpi）", flush=True)


def blank_note(ax, text: str):
    ax.imshow(np.full((RENDER, RENDER, 3), 245, np.uint8), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb"), sp.set_linewidth(0.6)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="#555555")


if __name__ == "__main__":
    main()
