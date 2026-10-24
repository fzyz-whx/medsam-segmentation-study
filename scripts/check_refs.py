# -*- coding: utf-8 -*-
"""refs.bib 校验脚本（幂等）：生成临时 aux 引用全部 key，调用 bibtex 真实解析，报告错误数。"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper"
BIB = PAPER / "refs.bib"

s = BIB.read_text(encoding="utf-8")
keys = [k.strip() for _, k in re.findall(r"@(\w+)\{([^,]+),", s)]
aux_lines = [
    r"\bibstyle{plain}",
    r"\citation{" + ",".join(keys) + "}",
    r"\bibdata{refs}",
]
aux = PAPER / "_check.aux"
aux.write_text("\n".join(aux_lines) + "\n", encoding="utf-8")

r = subprocess.run(
    ["bibtex", "_check"], cwd=str(PAPER), capture_output=True, text=True, encoding="utf-8", errors="replace"
)
print(r.stdout)
print(r.stderr)
print("bibtex_exit =", r.returncode)

blg = (PAPER / "_check.blg").read_text(encoding="utf-8", errors="replace") if (PAPER / "_check.blg").exists() else ""
errors = [ln for ln in blg.splitlines() if "error" in ln.lower()]
warns = [ln for ln in blg.splitlines() if "warning" in ln.lower()]
print("blg_errors =", len(errors))
for ln in errors:
    print("  ERR:", ln)
print("blg_warnings =", len(warns))
for ln in warns:
    print("  WARN:", ln)

bbl = PAPER / "_check.bbl"
n_entries = len(re.findall(r"\\bibitem", bbl.read_text(encoding="utf-8", errors="replace"))) if bbl.exists() else 0
print("bbl_bibitems =", n_entries)

for f in (aux, PAPER / "_check.blg", bbl):
    if f.exists():
        f.unlink()

sys.exit(0 if r.returncode == 0 and not errors else 1)
