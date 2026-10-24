#!/usr/bin/env bash
# 实验B 一键执行：2 数据集 x 3 档位（幂等，已完成自动跳过）→ 聚合 → 学习曲线图
# 用法：bash scripts/run_experiment_B.sh > logs/run_experiment_B.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
mkdir -p logs results/unet_runs figures checkpoints

# full 档位 n = 训练池大小 = 总训练集去掉末 1/10 验证集
#   Kvasir-SEG: 800 - 80  = 720
#   ISIC2018:   2075 - 207 = 1868
# （与 scripts/unet.py resolve_tier 的独立计算互为校验，不一致时 python 侧会报错）
run () {  # run <dataset> <tier>
  local ds="$1" tier="$2" n
  case "$ds" in
    Kvasir-SEG) if [ "$tier" = full ]; then n=720; else n=$tier; fi ;;
    ISIC2018)   if [ "$tier" = full ]; then n=1868; else n=$tier; fi ;;
  esac
  local log="logs/unet_${ds}_${n}.log"
  echo "[runB $(date '+%F %T')] START $ds tier=$tier (n=$n) -> $log"
  "$PY" scripts/unet.py --dataset "$ds" --tier "$tier" > "$log" 2>&1
  echo "[runB $(date '+%F %T')] EXIT=$? $ds n=$n"
}

run Kvasir-SEG 20
run Kvasir-SEG 100
run Kvasir-SEG full
run ISIC2018 20
run ISIC2018 100
run ISIC2018 full

"$PY" scripts/unet.py --aggregate-only > logs/unet_aggregate.log 2>&1
echo "[runB $(date '+%F %T')] aggregate EXIT=$?"
"$PY" scripts/unet.py --figure-only > logs/unet_figure.log 2>&1
echo "[runB $(date '+%F %T')] figure EXIT=$?"
echo "[runB $(date '+%F %T')] ALL DONE"
