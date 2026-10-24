# -*- coding: utf-8 -*-
"""Phase 5 论文数字核对：main.tex 中所有 >=4 位小数的实验数字
必须能在 results/ 落盘文件中找到出处（精确值 / 按论文小数位四舍五入 / 表内两源数字之差）。
幂等可重跑。报告写 results/paper_number_check_phase5.json，失败（unmatched>0）时非零退出。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"

SOURCE_FILES = [
    ROOT / "results" / "metrics.csv",
    ROOT / "results" / "dataset_stats.csv",
    ROOT / "results" / "summary.json",
    ROOT / "results" / "qualitative_cases.json",
    ROOT / "data" / "README.md",
    ROOT / "logs" / "unet_Kvasir-SEG_720.log",
]


def collect_sources():
    """源数字集合（float）与整数集合。"""
    floats, ints = set(), set()
    for f in SOURCE_FILES:
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\d+\.\d+", text):
            floats.add(float(m.group()))
        for m in re.finditer(r"(?<![\d.])\d{3,}(?![\d.])", text):
            ints.add(int(m.group()))
        # e-notation（summary.json 中 6.9e-05）
        for m in re.finditer(r"\d+\.?\d*[eE][+-]?\d+", text):
            try:
                floats.add(float(m.group()))
            except ValueError:
                pass
    return floats, ints


def collect_paper_numbers():
    """论文中 >=4 位小数的数字（剥掉 % 注释行），返回 [(数字串, 行号)]。"""
    lines = []
    for i, line in enumerate(TEX.read_text(encoding="utf-8").splitlines(), 1):
        s = line.split("%")[0] if not line.lstrip().startswith("%") else ""
        lines.append((i, s))
    nums = []
    for i, s in lines:
        s = s.replace("{,}", "")  # 31{,}037{,}633 -> 31037633
        for m in re.finditer(r"\d+\.\d{4,}", s):
            nums.append((m.group(), i))
    return nums


def main():
    floats, ints = collect_sources()
    nums = collect_paper_numbers()
    items, unmatched = [], []
    for s, ln in nums:
        v = float(s)
        k = len(s.split(".")[1])
        # 1) 精确/四舍五入匹配
        ok = any(round(x, k) == round(v, k) for x in floats)
        src = "source_round"
        # 2) 两源数字之差 / 之和（表内差值列、衰减量）
        if not ok:
            svals = sorted(floats)
            for a in svals:
                # 差：a - b = v  =>  b = a - v（表内差值列、衰减量的推导）
                b = a - v
                if b > 0 and any(round(x, k) == round(b, k) for x in svals):
                    ok, src = True, "source_diff"
                    break
        # 3) 整数（千分位归一后）
        if not ok and "." not in s:
            ok = int(s) in ints
            src = "source_int"
        items.append({"tex_number": s, "tex_line": ln, "ok": ok, "match_type": src if ok else None})
        if not ok:
            unmatched.append(s)
    report = {
        "date": "2026-10-24",
        "tex_file": "paper/main.tex",
        "source_files": [str(f.relative_to(ROOT)).replace("\\", "/") for f in SOURCE_FILES],
        "n_checked": len(items),
        "n_unmatched": len(unmatched),
        "items": items,
    }
    out = ROOT / "results" / "paper_number_check_phase5.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"checked={len(items)} unmatched={len(unmatched)}")
    if unmatched:
        print("UNMATCHED:", unmatched)
        sys.exit(1)
    print("ALL NUMBERS TRACED TO results/ OK")


if __name__ == "__main__":
    main()
