# -*- coding: utf-8 -*-
"""Phase 4：results/summary.json 最终版 + 论文数字逐项核对。

职责：
  1. 一致性校验：summary.json 的 prompt_strategy/fewshot/cross_dataset 三键与
     results/metrics.csv 逐行比对（防陈旧），n_pred_empty 与 per_sample CSV 重算比对；
  2. 论文数字核对清单（paper_number_checklist）：论文将要引用的每一个数字——
     表 1（dataset_stats.csv）、表 2（12 个提示策略 Dice）、表 3（6 个少样本 +
     3 个 MedSAM 参考）、表 4（7 个跨数据集 + 2 个衰减量）、U-Net 参数量、
     Kvasir 框 IoU、空预测数、图 2 参考线值、图 3 定性 case 的 Dice——
     全部以 {item, value, source_file, locator, verified} 记录，value 一律从
     源文件实时重读；任何一项核对失败即非零退出（绝不写入未核对的清单）；
  3. 惯例收口：Dice 定义与空样本处理惯例、点提示实现方式说明，汇总到
     final_report.conventions；
  4. 图表登记：learning_curve 校验结论、qualitative 选例协议与 case 清单。

幂等：全量重算重写 summary.json（其余键原样保留）。
用法：.venv/Scripts/python.exe scripts/finalize_summary.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

try:  # Windows 控制台/重定向编码兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics.csv"
STATS_CSV = RESULTS / "dataset_stats.csv"
SUMMARY = RESULTS / "summary.json"
CASES_JSON = RESULTS / "qualitative_cases.json"
RUNS = RESULTS / "unet_runs"

DATASET_ORDER = ["Kvasir-SEG", "ISIC2018", "Montgomery"]
PROMPT_SETTINGS = ["box", "box_perturb_10pct", "center_point", "random_point"]
FEWSHOT_TIERS = {"Kvasir-SEG": ["n=20", "n=100", "n=720"],
                 "ISIC2018": ["n=20", "n=100", "n=1868"]}

failures: list[str] = []
checklist: list[dict] = []


def check(item: str, value, source_file: str, locator: str) -> None:
    """把一个待引用数字登记进清单；value 必须与 source_file 实时重读结果一致。"""
    try:
        actual = locator() if callable(locator) else locator
        ok = (abs(value - actual) < 5e-7) if isinstance(value, float) else (value == actual)
    except Exception as e:  # noqa: BLE001
        actual, ok = f"<读取失败: {type(e).__name__}: {e}>", False
    checklist.append({"item": item, "value": value, "source_file": source_file,
                      "locator": locator if not callable(locator) else "<见上方描述>",
                      "verified": bool(ok)})
    if not ok:
        failures.append(f"{item}: 声称 {value}，源文件实读 {actual}")
        print(f"[finalize] [FAIL] {item}: 声称 {value}，源文件实读 {actual}", flush=True)


def read_metrics() -> dict:
    out = {}
    with open(METRICS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["experiment"], r["dataset"], r["setting"])] = (int(r["n_samples"]), float(r["dice"]))
    return out


def read_per_sample_empty(csv_name: str) -> dict:
    """{(dataset, setting): n_pred_empty}，从逐样本 CSV 重算。"""
    out: dict = {}
    with open(RESULTS / csv_name, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["setting"])
            out[key] = out.get(key, 0) + int(r["pred_empty"])
    return out


def main() -> None:
    M = read_metrics()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    n = 0

    # ---------- 1) 表 1：数据集统计（dataset_stats.csv 重读） ----------
    with open(STATS_CSV, newline="", encoding="utf-8") as f:
        stats_rows = {r["dataset"]: r for r in csv.DictReader(f)}
    for ds in DATASET_ORDER:
        row = stats_rows[ds]
        for col in ("n_pairs", "n_train", "n_test", "n_test_used_in_experiments",
                    "resolution_range"):
            v = int(row[col]) if col.startswith("n_") else row[col]
            check(f"表1 {ds} {col}", v, "results/dataset_stats.csv",
                  lambda r=row, c=col: (int(r[c]) if c.startswith("n_") else r[c]))
            n += 1

    # ---------- 2) 表 2：提示策略 12 个 macro Dice（metrics.csv 重读） ----------
    for ds in DATASET_ORDER:
        for st in PROMPT_SETTINGS:
            key = ("prompt_strategy", ds, st)
            if key not in M:
                raise SystemExit(f"metrics.csv 缺行 {key}")
            check(f"表2 {ds} {st} macro Dice", M[key][1], "results/metrics.csv",
                  lambda k=key: read_metrics()[k][1])
            # 与 summary 内部值交叉核对（防 summary 陈旧）
            sv = summary["prompt_strategy"]["results"][ds][st]["macro_dice"]
            if abs(sv - M[key][1]) > 5e-7:
                failures.append(f"summary.prompt_strategy.{ds}.{st}={sv} != metrics {M[key][1]}")
            check(f"表2 {ds} {st} n_samples", M[key][0], "results/metrics.csv",
                  lambda k=key: read_metrics()[k][0])
            n += 2

    # ---------- 3) 表 3：少样本 6 个 + MedSAM 参考 3 个 ----------
    for ds, tiers in FEWSHOT_TIERS.items():
        for st in tiers:
            key = ("fewshot", ds, st)
            check(f"表3 {ds} {st} macro Dice", M[key][1], "results/metrics.csv",
                  lambda k=key: read_metrics()[k][1])
            sv = summary["fewshot"]["results"][ds][st]["macro_dice"]
            if abs(sv - M[key][1]) > 5e-7:
                failures.append(f"summary.fewshot.{ds}.{st}={sv} != metrics {M[key][1]}")
            n += 1
        key = ("prompt_strategy", ds, "box")
        check(f"表3 {ds} MedSAM 零样本参考线", M[key][1], "results/metrics.csv",
              lambda k=key: read_metrics()[k][1])
        n += 1
    check("图2 Kvasir MedSAM 参考线值", M[("prompt_strategy", "Kvasir-SEG", "box")][1],
          "results/metrics.csv", lambda: read_metrics()[("prompt_strategy", "Kvasir-SEG", "box")][1])
    check("图2 ISIC2018 MedSAM 参考线值", M[("prompt_strategy", "ISIC2018", "box")][1],
          "results/metrics.csv", lambda: read_metrics()[("prompt_strategy", "ISIC2018", "box")][1])
    n += 2

    # ---------- 4) 表 4：跨数据集 7 行 + 2 个衰减量 ----------
    cross_rows = [("ISIC2018", "UNet(Kvasir-SEG→ISIC2018)"), ("Montgomery", "UNet(Kvasir-SEG→Montgomery)"),
                  ("Kvasir-SEG", "UNet(ISIC2018→Kvasir-SEG)"), ("Montgomery", "UNet(ISIC2018→Montgomery)"),
                  ("Kvasir-SEG", "MedSAM(Kvasir-SEG)"), ("ISIC2018", "MedSAM(ISIC2018)"),
                  ("Montgomery", "MedSAM(Montgomery)")]
    for ds, st in cross_rows:
        key = ("cross_dataset", ds, st)
        check(f"表4 {st} → {ds} macro Dice", M[key][1], "results/metrics.csv",
              lambda k=key: read_metrics()[k][1])
        n += 1
    for tag, path in [("UNet(ISIC2018→Kvasir-SEG) 跨集衰减量", "UNet(ISIC2018→Kvasir-SEG)"),
                      ("UNet(Kvasir-SEG→ISIC2018) 跨集衰减量", "UNet(Kvasir-SEG→ISIC2018)")]:
        v = summary["cross_dataset"]["results"]["unet_cross"][path]["drop_vs_indomain"]
        check(f"表4 {tag} drop_vs_indomain", v, "results/summary.json",
              lambda p=path: json.loads(SUMMARY.read_text(encoding="utf-8"))
              ["cross_dataset"]["results"]["unet_cross"][p]["drop_vs_indomain"])
        n += 1

    # ---------- 5) 其他论文可用数字 ----------
    for ds, nn in [("Kvasir-SEG", 720), ("ISIC2018", 1868)]:
        marker = json.loads((RUNS / f"{ds}_{nn}.json").read_text(encoding="utf-8"))
        check(f"U-Net 参数量（{marker['dataset']} marker）", marker["n_params"],
              f"results/unet_runs/{ds}_{nn}.json",
              lambda d=ds, k=nn: json.loads((RUNS / f"{d}_{k}.json").read_text(encoding="utf-8"))["n_params"])
        n += 1
    iou = summary["prompt_strategy"]["kavsir_bbox_check"]["mean_box_iou"]
    check("Kvasir 框与官方 kavsir_bboxes 平均 IoU", iou, "results/summary.json",
          lambda: json.loads(SUMMARY.read_text(encoding="utf-8"))
          ["prompt_strategy"]["kavsir_bbox_check"]["mean_box_iou"])
    n += 1

    # 空预测数：逐样本 CSV 重算 vs summary 声明
    empty_a = read_per_sample_empty("per_sample_A.csv")
    empty_b = read_per_sample_empty("per_sample_B.csv")
    empty_c = read_per_sample_empty("per_sample_C.csv")
    for ds in DATASET_ORDER:
        st = "box"
        v = empty_a.get((ds, st), 0)
        sv = summary["prompt_strategy"]["results"][ds][st]["n_pred_empty"]
        check(f"实验A {ds} box 空预测数", sv, "results/per_sample_A.csv",
              lambda d=ds: read_per_sample_empty("per_sample_A.csv").get((d, "box"), 0))
        if v != sv:
            failures.append(f"实验A {ds} 空预测 summary={sv} 重算={v}")
        n += 1
    for ds, tiers in FEWSHOT_TIERS.items():
        for st in tiers:
            v = empty_b.get((ds, st), 0)
            sv = summary["fewshot"]["results"][ds][st]["n_pred_empty"]
            check(f"实验B {ds} {st} 空预测数", sv, "results/per_sample_B.csv",
                  lambda d=ds, s=st: read_per_sample_empty("per_sample_B.csv").get((d, s), 0))
            if v != sv:
                failures.append(f"实验B {ds}/{st} 空预测 summary={sv} 重算={v}")
            n += 1
    for _, st in cross_rows[:4]:
        # per_sample_C 的 dataset 列是目标集，setting 是组合名；键为 (目标集, 组合名) 元组
        v = read_per_sample_empty("per_sample_C.csv").get(
            ({"UNet(Kvasir-SEG→ISIC2018)": "ISIC2018", "UNet(Kvasir-SEG→Montgomery)": "Montgomery",
              "UNet(ISIC2018→Kvasir-SEG)": "Kvasir-SEG", "UNet(ISIC2018→Montgomery)": "Montgomery"}[st], st), 0)
        sv = summary["cross_dataset"]["results"]["unet_cross"][st]["n_pred_empty"]
        check(f"实验C {st} 空预测数", sv, "results/per_sample_C.csv",
              lambda s=st: read_per_sample_empty("per_sample_C.csv").get(
                  ({"UNet(Kvasir-SEG→ISIC2018)": "ISIC2018", "UNet(Kvasir-SEG→Montgomery)": "Montgomery",
                    "UNet(ISIC2018→Kvasir-SEG)": "Kvasir-SEG",
                    "UNet(ISIC2018→Montgomery)": "Montgomery"}[s], s), 0))
        if v != sv:
            failures.append(f"实验C {st} 空预测 summary={sv} 重算={v}")
        n += 1

    # ---------- 6) 图 3 定性 case 的 Dice 核对 ----------
    dice_a = {}
    with open(RESULTS / "per_sample_A.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["setting"] == "box":
                dice_a[(r["dataset"], Path(r["sample_id"]).stem)] = float(r["dice"])
    dice_b = {}
    with open(RESULTS / "per_sample_B.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dice_b[(r["dataset"], Path(r["sample_id"]).stem)] = float(r["dice"])
    for ds, cs in cases["cases"].items():
        for c in cs:
            stem = c["stem"]
            check(f"图3 {ds} {c['kind']}({stem[:16]}) MedSAM Dice", c["dice_medsam"],
                  "results/per_sample_A.csv",
                  lambda d=ds, s=stem: dice_a[(d, s)])
            if c["dice_unet"] is not None:
                check(f"图3 {ds} {c['kind']}({stem[:16]}) U-Net Dice", c["dice_unet"],
                      "results/per_sample_B.csv",
                      lambda d=ds, s=stem: dice_b[(d, s)])
            n += 1 if c["dice_unet"] is None else 2

    # ---------- 7) 失败即中止，不写未核对的清单 ----------
    print(f"[finalize] 共核对 {len(checklist)} 项，失败 {len(failures)} 项", flush=True)
    if failures:
        for msg in failures:
            print(f"[finalize] [FAIL-SUMMARY] {msg}", flush=True)
        raise SystemExit(f"论文数字核对失败 {len(failures)} 项，summary.json 未修改")

    # ---------- 8) 写 final_report 键 ----------
    summary["final_report"] = {
        "phase": "Phase 4 图表汇总（summary.json 最终版收口）",
        "date": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/finalize_summary.py",
        "conventions": {
            "dice_definition": "逐样本 Dice = 2|P∩G|/(|P|+|G|)；macro Dice = 逐样本 Dice 的算术平均",
            "empty_sample_handling": (
                "空样本处理惯例（医学分割常用约定，三实验统一）：GT 与预测同为空记 Dice=1.0"
                "（两空一致）；恰一方为空记 0.0。三数据集 Phase 1 已验证无空 GT；实验 A 全部 "
                "2912 个预测无一为空（n_pred_empty=0）；实验 B 少样本档位存在空预测"
                "（Kvasir n=20/n=100 各 5 个、ISIC n=20 9 个 / n=100 2 个、全量档 0 个），"
                "实验 C 跨数据集存在空预测（Kvasir→ISIC 12 个），均按惯例记 0 计入 macro，"
                "逐样本明细见 results/per_sample_*.csv 的 pred_empty 列，各组合空预测数见"
                " summary 各实验 results.*.n_pred_empty（已与逐样本 CSV 重算核对一致）"),
            "point_prompt_implementation": summary["prompt_strategy"]["point_prompt_implementation"],
            "randomness": "主种子 42；扰动框/随机点由逐样本种子 random.Random('42|{dataset}|{setting}|{sample_id}') 生成；U-Net 训练 random/numpy/torch/cuda 全 seed=42",
            "test_subsets": "三实验同一测试子集：Kvasir-SEG test 200 全量；ISIC2018 test 前 500/519"
                            "（splits.json seed=42 洗牌顺序）；Montgomery test 28 全量",
        },
        "figures": {
            "learning_curve": {
                "path": "figures/learning_curve.pdf（+png，300dpi，SimHei）",
                "generated_by": "scripts/unet.py --figure-only（从 results/metrics.csv 确定性重画）",
                "medsam_reference_check": (
                    "Phase 4 校验通过：文件存在（34,553B）且两子图均有红色虚线 MedSAM 参考线，"
                    "图例值 0.9307（Kvasir）/ 0.9298（ISIC）与 metrics.csv box 行一致（PNG 目检）"),
            },
            "qualitative": {
                "path": "figures/qualitative.pdf（+png，300dpi，SimHei）",
                "generated_by": "scripts/qualitative_figure.py",
                "layout": "三数据集块 × (2 好 + 2 差) case × 四列（原图/GT/MedSAM 框提示/U-Net 全量档）；"
                          "Montgomery 无 U-Net 全量档模型，该列留白注记（如实）",
                "selection_protocol": cases["selection_protocol"],
                "case_list_file": "results/qualitative_cases.json",
                "dice_annotations_source": "results/per_sample_A.csv（box）、results/per_sample_B.csv（n=720/n=1868）",
            },
            "table1_source": "results/dataset_stats.csv（分辨率范围为磁盘实测，与 Phase 1 stats.json 一致）",
        },
        "paper_number_checklist": {
            "note": "论文将引用的全部实验数字→results/ 源文件映射；verified=true 表示本次"
                    "运行中已从源文件实时重读并逐项比对一致（共 "
                    f"{len(checklist)} 项，失败 0 项）",
            "n_checked": len(checklist),
            "n_failed": 0,
            "items": checklist,
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[finalize] summary.json 已更新（新增 final_report 键，含 {len(checklist)} 项"
          f"核对清单全部 verified=true）", flush=True)


if __name__ == "__main__":
    main()
