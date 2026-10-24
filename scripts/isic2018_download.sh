#!/usr/bin/env bash
# ISIC 2018 Task1 downloader: ModelScope OmniData/ISIC_2018_Task_1 official tar.gz
# Resumable loop: curl -C - with slow-speed abort, retries until complete.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/ISIC2018/_download logs
URL="https://modelscope.cn/datasets/OmniData/ISIC_2018_Task_1/resolve/master/raw/ISIC_2018_Task_1.tar.gz"
OUT="data/ISIC2018/_download/ISIC_2018_Task_1.tar.gz"
EXPECT=13799882093
echo "=== $(date) start download ==="
for i in $(seq 1 300); do
  SIZE=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$SIZE" -ge "$EXPECT" ]; then
    echo "DONE size=$SIZE expected=$EXPECT"
    exit 0
  fi
  echo "[attempt $i] have=$SIZE / $EXPECT, resuming..."
  curl -L -C - --retry 5 --retry-delay 3 --connect-timeout 30 \
       --speed-time 60 --speed-limit 10240 \
       -o "$OUT" "$URL"
  sleep 2
done
echo "FAILED: not complete after 300 attempts"
exit 1
