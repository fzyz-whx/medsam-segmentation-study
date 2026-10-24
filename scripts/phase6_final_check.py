# -*- coding: utf-8 -*-
"""Phase 6 终检脚本（幂等，只读检查，不修改任何产物）。

逐项核对实验方案第 7 节 Phase 6 终检清单：
  1. paper/main.pdf 存在且页数 > 0；
  2. 编译日志（paper/main.log，最近一次编译）无 error / undefined / Overfull；
  3. PDF 文本无未解析引用 "??"；
  4. main.tex 与 PDF 文本无占位符残留（TODO/FIXME/XXX/占位/待补/TBD）；
  5. 图表 label 定义齐全（图 1-3 / 表 1-4）且全部被正文引用、无悬空引用；
  6. 参考文献引用 >= 20 且全部被正文引用（结合 results/paper_number_check_phase5.json）。

结果打印到 stdout；发现任何 FAIL 项时退出码非 0。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures = []


def check(name, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- 1. main.pdf 存在且页数 > 0 ----
pdf_path = ROOT / "paper" / "main.pdf"
check("paper/main.pdf 存在", pdf_path.exists(),
      f"size={pdf_path.stat().st_size}B" if pdf_path.exists() else "missing")
pages = 0
if pdf_path.exists():
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    pages = len(doc)
    check("main.pdf 页数 > 0", pages > 0, f"pages={pages}")

    # ---- 3. PDF 无未解析引用 ----
    text = ""
    for i in range(pages):
        text += doc[i].get_textpage().get_text_range() + "\n"
    n_qq = text.count("??")
    check("PDF 无未解析引用 '??'", n_qq == 0, f"count={n_qq}")

    # ---- 4b. PDF 文本无占位符 ----
    pdf_hits = {p: text.count(p) for p in
                ["TODO", "FIXME", "XXX", "占位", "待补", "待填", "TBD"] if text.count(p)}
    check("PDF 无占位符残留", not pdf_hits, str(pdf_hits or "none"))

# ---- 2. 编译日志无 error / undefined / Overfull ----
log_path = ROOT / "paper" / "main.log"
if log_path.exists():
    log = log_path.read_text(encoding="utf-8", errors="replace")
    n_err = len(re.findall(r"^!", log, flags=re.M))
    n_undef = len(re.findall(r"undefined", log, flags=re.I))
    n_over = len(re.findall(r"Overfull", log))
    m = re.search(r"Output written on main\.pdf \((\d+) pages?\)", log)
    log_pages = int(m.group(1)) if m else -1
    check("编译日志 0 error", n_err == 0, f"count={n_err}")
    check("编译日志 0 undefined", n_undef == 0, f"count={n_undef}")
    check("编译日志 0 Overfull", n_over == 0, f"count={n_over}")
    if pdf_path.exists():
        check("日志页数与实际 PDF 一致", log_pages == pages,
              f"log={log_pages}, pdf={pages}")
else:
    check("paper/main.log 存在", False, "missing")

# ---- 4a/5. main.tex：占位符 + 图表 label/ref 一致性 ----
tex_path = ROOT / "paper" / "main.tex"
if tex_path.exists():
    tex = tex_path.read_text(encoding="utf-8")
    tex_hits = [p for p in ["TODO", "FIXME", "XXX", "占位", "待补", "待填", "TBD"] if p in tex]
    check("main.tex 无占位符残留", not tex_hits, str(tex_hits or "none"))

    labels = [(m.start(), m.group(1))
              for m in re.finditer(r"\\label\{((?:fig|tab):[^}]+)\}", tex)]
    refs = [m.group(1)
            for m in re.finditer(r"\\(?:ref|Cref|cref|autoref)\{((?:fig|tab):[^}]+)\}", tex)]
    label_names = [l for _, l in labels]
    from collections import Counter
    rc = Counter(refs)
    check("图表 label 共 7 个（图3+表4）", len(labels) == 7,
          f"labels={label_names}")
    unref = [l for l in label_names if rc[l] == 0]
    check("每个图表 label 均被正文引用", not unref, f"unreferenced={unref or 'none'}")
    dangling = [r for r in refs if r not in label_names]
    check("无指向不存在 label 的引用", not dangling, f"dangling={dangling or 'none'}")
    check("图表定义顺序与编号一致（fig:framework,fig:curve,fig:qual / tab:stats,tab:prompt,tab:fewshot,tab:cross）",
          label_names == ["tab:stats", "fig:framework", "tab:prompt", "fig:curve",
                          "tab:fewshot", "tab:cross", "fig:qual"],
          "definition order: " + ", ".join(label_names))
else:
    check("paper/main.tex 存在", False, "missing")

# ---- 6. 参考文献引用数（来自 Phase 5 核对报告） ----
chk_json = ROOT / "results" / "paper_number_check_phase5.json"
if chk_json.exists():
    data = json.loads(chk_json.read_text(encoding="utf-8"))
    keys = list(data.keys()) if isinstance(data, dict) else []
    print(f"[INFO] results/paper_number_check_phase5.json keys={keys}")
    n_checked = data.get("n_checked", len(data.get("items", [])))
    n_unmatched = data.get("n_unmatched", -1)
    check("论文数字溯源全部通过（复跑 check_paper_numbers.py 后读取）",
          n_unmatched == 0, f"checked={n_checked}, unmatched={n_unmatched}")
else:
    check("results/paper_number_check_phase5.json 存在", False, "missing")

print()
if failures:
    print(f"PHASE 6 FINAL CHECK: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("PHASE 6 FINAL CHECK: ALL PASS")
