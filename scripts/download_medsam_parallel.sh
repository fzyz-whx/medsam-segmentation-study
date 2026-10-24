#!/usr/bin/env bash
# MedSAM ViT-B 权重并行分片下载（Zenodo 官方 DOI 存档 10689643，幂等可重跑）
# 每个分片单独 curl + 续传，完成分片按字节数校验；全部完成后拼接 + md5 校验。
set -u
cd "$(dirname "$0")/.."
URL="https://zenodo.org/api/records/10689643/files/medsam_vit_b.pth/content"
OUT="checkpoints/medsam_vit_b.pth"
TOTAL=375049145
EXPECT_MD5="3bb6db55bd0c9ca30b61248bca72f8d6"
N=16
CHUNK=$(( (TOTAL + N - 1) / N ))
PARTDIR="checkpoints/.medsam_parts"
mkdir -p "$PARTDIR" checkpoints

if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT" 2>/dev/null || echo 0)" = "$TOTAL" ]; then
  md5=$(md5sum "$OUT" | awk '{print $1}')
  if [ "$md5" = "$EXPECT_MD5" ]; then echo "already complete"; exit 0; fi
  rm -f "$OUT"
fi

dl_part() {
  local i=$1 start=$2 end=$3
  local pf="$PARTDIR/part_$(printf '%02d' "$i")"
  local want=$(( end - start + 1 ))
  for attempt in 1 2 3 4 5 6; do
    local have; have=$(stat -c %s "$pf" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && return 0
    # 续传：从已有字节后继续
    local from=$(( start + have ))
    curl -s --noproxy '*' --connect-timeout 30 --speed-time 45 --speed-limit 8192 \
         -r "$from-$end" -o - "$URL" >> "$pf" 2>/dev/null
  done
  have=$(stat -c %s "$pf" 2>/dev/null || echo 0)
  [ "$have" = "$want" ]
}

echo "[$(date '+%F %T')] launching $N parallel chunk downloads (chunk=${CHUNK}B)"
pids=()
for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  dl_part "$i" "$start" "$end" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[$(date '+%F %T')] chunks done (fail=$fail), verifying sizes..."
ok=1
for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  want=$(( end - start + 1 ))
  pf="$PARTDIR/part_$(printf '%02d' "$i")"
  have=$(stat -c %s "$pf" 2>/dev/null || echo 0)
  if [ "$have" != "$want" ]; then echo "part $i: $have/$want MISMATCH"; ok=0; fi
done
if [ "$ok" = 1 ]; then
  cat "$PARTDIR"/part_* > "$OUT"
  size=$(stat -c %s "$OUT")
  md5=$(md5sum "$OUT" | awk '{print $1}')
  echo "assembled size=$size md5=$md5"
  if [ "$size" = "$TOTAL" ] && [ "$md5" = "$EXPECT_MD5" ]; then
    echo "MD5_OK"
    rm -rf "$PARTDIR"
    exit 0
  fi
fi
echo "INCOMPLETE (some parts missing) — rerun this script to resume"
exit 1
