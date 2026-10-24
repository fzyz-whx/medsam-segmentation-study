# data/ — 数据集总览（Phase 1 验收）

下载日期：全部为 **2026-10-24**。下载细节见各数据集目录内 `DOWNLOAD_LOG.md`；机器可读统计见各 `stats.json`，划分为 `splits.json`（均 **8:2，seed=42**，幂等可复现）。Phase 1 收口由 `scripts/phase1_recheck.py` 独立复核，报告落盘 `results/phase1_recheck.json`（2026-10-24 全项 PASS）。

| 数据集 | 模态/任务 | 图-mask 对 | train/test (8:2) | 主要来源（实际下载） |
|---|---|---|---|---|
| Kvasir-SEG | 内镜胃肠息肉分割 | 1000 | 800 / 200 | ModelScope 镜像 `OmniData/Kvasir-SEG`（官方 kvasir-seg.zip 原包，字节数一致 46,227,172） |
| ISIC2018 Task1 | 皮肤镜病变分割 | 2594 | 2075 / 519 | ModelScope 镜像 `OmniData/ISIC_2018_Task_1`（官方 5 个 zip 的原包 tar.gz，13,799,820,893 B，sha256 与 API 声明一致） |
| Montgomery | 胸片肺分割 | 138 | 110 / 28 | hf-mirror `Famatsu123/montgomery-shenzhen-tuberculosis-cxr` 仓库内 MontgomerySet.zip（NLM 原始目录结构镜像，616,701,553 B，sha256 校验通过） |

## 1. Kvasir-SEG（`data/Kvasir-SEG/`）

- **内容**：胃肠息肉分割（Kvasir-SEG，Jha et al. 2020），1000 对图像+mask，官方声明 1000，一致。
- **获取**：官方源 datasets.simula.no 为境外（留最后手段）；GitHub simula/kvasir-seg 仓库已迁移至 DebeshJha/Kvasir-SEG 且无 release；hf-mirror 无镜像 → 按镜像优先级取 **ModelScope `OmniData/Kvasir-SEG` raw/kvasir-seg.zip**（官方原包镜像，curl 国内直连 6.8 MB/s）。
- **结构**：`images/`（1000 jpg）+ `masks/`（1000 jpg，与图像同名同尺寸）+ `kavsir_bboxes.json`（1000 条 GT 外接框，实验 A 框提示直接可用）+ `splits.json` + `stats.json`。原包缓存 `data/kvasir-seg.zip` 保留。
- **统计摘要**：尺寸不统一（333 种变体，主流 622x530、626x547 等）；mask 为 RGB JPEG，灰度化后值域 [0,255]，JPEG 边界伪影中间灰度占比 mean 0.77% / max 2.12% → 约定 **≥128 二值化**，二值化后无空 GT；逐对同尺寸 1000/1000；全量 PIL 可读。
- **处理脚本**：`scripts/prepare_kvasir_seg.py`（幂等）。

## 2. ISIC2018 Task 1（`data/ISIC2018/`）

- **内容**：ISIC 2018 挑战赛 Task 1 皮肤病变分割训练集，2594 对（jpg 图 + png GT mask），与官方声明 2594 一致。
- **获取**：hf-mirror 候选仓库均为分类 parquet（无 mask）或全分辨率 parquet 但仅 284 KB/s（26h ETA）不可用 → **ModelScope `OmniData/ISIC_2018_Task_1`**（官方 zip 逐字节镜像 tar.gz）。8 段 Range 并行 + 断点续传下载，sha256 `506bacda…e5eb87a` 与 ModelScope API 声明完全一致。源头存档保留于 `_download/ISIC_2018_Task_1.tar.gz`。
- **结构**：`images/`（2594 jpg）+ `masks/`（2594 `*_segmentation.png`）+ `splits.json` + `stats.json` + `ATTRIBUTION.txt`/`ATTRIBUTION_GT.txt`/`LICENSE.txt`（官方 Input/GT 版权文件；收口时已将混入 masks/ 的 GT 版声明移出，masks/ 现为纯 2594 png）。
- **统计摘要**：206 种变分辨率（576x540 ~ 6748x4499）；mask 全为 L 模式严格二值 {0,255}；图-mask 逐对同尺寸 2594/2594；全量 PIL verify 可读 + 抽样 60 对全解码通过。磁盘保留原始分辨率，实验期按模型 resize。
- **处理脚本**：`scripts/isic2018_download_seg.sh`（下载，幂等可续）、`scripts/isic2018_verify.py`（校验+划分+stats，幂等）。

## 3. Montgomery（`data/Montgomery/`）

- **内容**：NLM Montgomery County 胸片集（Jaeger et al. 2014），138 张 CXR + 手工左右肺 mask 并集。官方声明 138，一致。
- **获取**：GitHub 各候选仓库无 release 资产、ModelScope 无结果 → **hf-mirror `Famatsu123/montgomery-shenzhen-tuberculosis-cxr`** 的 MontgomerySet.zip（传输中断 2 次均断点续传恢复，sha256 `e1543a55…8e8392` 校验通过）。
- **结构**：`images/`（138 png，原图字节级拷贝）+ `masks/`（138 png，leftMask|rightMask 并集，严格 {0,255}）+ `raw/MontgomerySet/`（解压原始结构）+ `splits.json` + `stats.json`。镜像包坑位：mask 目录名为 leftMask/rightMask（非 left_mask）；CXR_png 与 mask 目录各混入 1 个 Thumbs.db，glob `*.png` 排除后恰 138。
- **统计摘要**：两种尺寸（4892x4020 横位 41 张 + 4020x4892 竖位 97 张），图-mask 逐对同尺寸 138/138，全量 L 模式二值可读，无需对齐重采样。
- **处理脚本**：`scripts/prepare_montgomery.py`（幂等）。

## 统一约定（实验期适用）

- **划分**：各集 `splits.json`，8:2，`seed=42`，train/test 无交集、并集覆盖全量（复核脚本已逐项验证）。
- **resize 方案**：磁盘保留原始分辨率；MedSAM 推理统一 resize 1024x1024（box/mask 坐标同步缩放），U-Net 训练统一 resize 256x256（mask 最近邻插值）。
- **mask 二值化**：ISIC/Montgomery 天然 {0,255}；Kvasir mask 为 RGB JPEG，读取后灰度化并 **≥128 二值化**（伪影占比 <2.12%，stats 有记录）。
- **Windows 坑位**：本仓库路径含中文，`cv2.imread` 全部失败——需 OpenCV 时必须 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)`；本次校验一律用 PIL。

## 权重（`checkpoints/medsam_vit_b.pth`，非 data/ 但属 Phase 1 验收项）

官方 MedSAM ViT-B 权重，375,049,145 字节，MD5 `3bb6db55bd0c9ca30b61248bca72f8d6`，来源 **Zenodo 官方 DOI 存档** `https://zenodo.org/records/10689643`（bowang-lab 作者官方发布，非 Google Drive）。`torch.load(weights_only=True)` 通过：314 张量，键前缀恰为 `image_encoder`/`prompt_encoder`/`mask_decoder`，首键 `image_encoder.neck.0.weight`。证据：`results/medsam_verify.json`、`results/phase1_recheck.json`（复核亲验）、`checkpoints/WEIGHTS.md`。未触发实验方案 5.2 原版 SAM 兜底。
