#!/usr/bin/env bash
# MedSAM ViT-B 权重下载（Zenodo 官方 DOI 存档，断点续传，幂等可重跑）
# 源: https://zenodo.org/records/10689643 (bowang-lab/MedSAM 官方权重存档)
# 期望: 375049145 bytes, md5=3bb6db55bd0c9ca30b61248bca72f8d6
set -u
cd "$(dirname "$0")/.."
URL="https://zenodo.org/api/records/10689643/files/medsam_vit_b.pth/content"
OUT="checkpoints/medsam_vit_b.pth"
EXPECT_SIZE=375049145
EXPECT_MD5="3bb6db55bd0c9ca30b61248bca72f8d6"
mkdir -p checkpoints

for attempt in 1 2 3 4 5; do
  echo "[$(date '+%F %T')] attempt $attempt: curl -C - (resume) ..."
  curl -L --noproxy '*' --connect-timeout 30 --speed-time 60 --speed-limit 10240 \
       -o "$OUT" -C - "$URL"
  rc=$?
  size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "[$(date '+%F %T')] curl exit=$rc, size=$size / $EXPECT_SIZE"
  if [ "$size" = "$EXPECT_SIZE" ]; then
    echo "size OK, verifying md5 ..."
    md5=$(md5sum "$OUT" | awk '{print $1}')
    echo "md5=$md5 (expect $EXPECT_MD5)"
    if [ "$md5" = "$EXPECT_MD5" ]; then
      echo "MD5_OK"
      exit 0
    else
      echo "MD5_MISMATCH, redownload from scratch"
      rm -f "$OUT"
    fi
  fi
  sleep 3
done
echo "DOWNLOAD_FAILED after 5 attempts"
exit 1
