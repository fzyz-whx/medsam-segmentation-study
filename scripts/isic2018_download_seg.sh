#!/usr/bin/env bash
# Segmented parallel downloader for ISIC_2018_Task_1.tar.gz (ModelScope mirror)
# 8 range segments, each resumable; then concatenate + verify size.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/ISIC2018/_download logs
URL="https://modelscope.cn/datasets/OmniData/ISIC_2018_Task_1/resolve/master/raw/ISIC_2018_Task_1.tar.gz"
OUT="data/ISIC2018/_download/ISIC_2018_Task_1.tar.gz"
TOTAL=13799882093
NSEG=8
SEG=$(( (TOTAL + NSEG - 1) / NSEG ))

download_part() {
  local idx=$1 start=$2 end=$3
  local f="data/ISIC2018/_download/part_$idx"
  local expect=$(( end - start + 1 ))
  local tries=0
  while [ "$tries" -lt 200 ]; do
    local have=0
    [ -f "$f" ] && have=$(stat -c %s "$f")
    if [ "$have" -ge "$expect" ]; then echo "part_$idx COMPLETE ($have/$expect)"; return 0; fi
    if [ "$have" -gt "$expect" ]; then echo "part_$idx OVERSIZE ($have/$expect), truncating"; truncate -s "$expect" "$f"; have=$expect; continue; fi
    local from=$(( start + have ))
    echo "part_$idx have=$have/$expect from=$from"
    curl -s -f -L --retry 5 --retry-delay 3 --connect-timeout 30 \
         --speed-time 45 --speed-limit 20480 \
         -r ${from}-${end} "$URL" >> "$f" || sleep 2
    tries=$(( tries + 1 ))
  done
  echo "part_$idx FAILED after $tries tries"
  return 1
}

echo "=== $(date) segmented download start, TOTAL=$TOTAL NSEG=$NSEG SEG=$SEG ==="
PIDS=()
for i in $(seq 0 $((NSEG-1))); do
  S=$(( i * SEG )); E=$(( (i+1) * SEG - 1 ))
  [ $E -ge $TOTAL ] && E=$(( TOTAL - 1 ))
  download_part "$i" "$S" "$E" > "logs/isic2018_part_$i.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then echo "SOME PARTS FAILED"; exit 1; fi

# concatenate
echo "=== concatenating ==="
rm -f "$OUT"
for i in $(seq 0 $((NSEG-1))); do cat "data/ISIC2018/_download/part_$i" >> "$OUT"; done
FSIZE=$(stat -c %s "$OUT")
echo "final size=$FSIZE expected=$TOTAL"
if [ "$FSIZE" -ne "$TOTAL" ]; then echo "SIZE MISMATCH"; exit 1; fi
echo "=== $(date) ALL_DONE ==="
