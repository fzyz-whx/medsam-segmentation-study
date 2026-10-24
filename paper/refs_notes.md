# refs.bib 核实笔记（文献核查阶段）

- 完成日期：2026-10-24
- 产物：`paper/refs.bib`，共 **26 条**（必引清单 10 条全数收录，自补 16 条），missing 为空。
- 核实方式：逐条通过 **Crossref API / arXiv abs 页 / CVF Open Access / PMC / PubMed E-utilities / 出版社（Springer、IEEE、Nature、Elsevier、MDPI、QIMS）官方元数据** 核对作者顺序、标题、venue、年份、卷期页/DOI，未凭记忆编写任何条目。
- 说明：DBLP（dblp.org，含 search API）当日被 Anubis 反爬整站拦截（返回 "Loading..." 人机校验页），故全部改用上述权威源核实；Nature 官网页面 303 跳转至 idp.nature.com，亦改用 Crossref 元数据（与 DOI 注册库一致）。

## 必引清单核实结果（10/10）

| # | BibTeX key | 条目 | 核实来源 |
|---|---|---|---|
| 1 | kirillov2023segment | Kirillov et al., "Segment Anything", ICCV 2023, pp. 4015–4026 | CVF: openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html |
| 2 | ma2024segment | Ma et al., "Segment Anything in Medical Images", Nat. Commun. 15(1):654, 2024, DOI 10.1038/s41467-024-44824-z | Crossref: api.crossref.org/works/10.1038/s41467-024-44824-z |
| 3 | ma2025medsam2 | Ma et al., "MedSAM2: Segment Anything in 3D Medical Images and Videos", arXiv:2504.03600, 2025（截至核实日无期刊 journal-ref，按 arXiv 引用） | arxiv.org/abs/2504.03600 |
| 4 | ronneberger2015unet | Ronneberger et al., "U-Net", MICCAI 2015 (LNCS), pp. 234–241, DOI 10.1007/978-3-319-24574-4_28 | Crossref |
| 5 | isensee2021nnunet | Isensee et al., "nnU-Net", Nature Methods 18(2):203–211, 2021 | Crossref: 10.1038/s41592-020-01008-z |
| 6 | koishiyeva2025review | Koishiyeva et al., "A Review of Deep Learning Approaches Based on SAM for Medical Image Segmentation", Bioengineering 12(12):1312, 2025（即实验方案 PMC12729286） | pmc.ncbi.nlm.nih.gov/articles/PMC12729286/ |
| 7 | gu2025how | Gu et al., "How to build the best medical image segmentation algorithm using foundation models...", MELBA (J. Mach. Learn. Biomed. Imaging) 2025:006, DOI 10.59275/j.melba.2025-86a6。**注意：实验方案猜测载于 Medical Image Analysis 2025；经 Crossref 按 MedIA 容器过滤检索无此文，arXiv journal-ref 与 Crossref 均确认正式出处为 MELBA，已按真实 venue 收录** | Crossref: 10.59275/j.melba.2025-86a6 + arxiv.org/abs/2404.09957 |
| 8 | jha2020kvasir | Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020 (LNCS), pp. 451–462, DOI 10.1007/978-3-030-37734-2_37 | Crossref |
| 9 | codella2019skin | Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC", arXiv:1902.03368（ISIC 2018 Task 1 数据来源） | arxiv.org/abs/1902.03368 |
| 10 | jaeger2014two | Jaeger et al., "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases", QIMS 4(6):475–477, 2014, DOI 10.3978/j.issn.2223-4292.2014.11.20。**注意：真实标题为 pulmonary diseases（实验方案误记 pneumothorax，以 PubMed 为准）** | PubMed PMID 25525580 |

## 自补条目核实结果（16 条）

| BibTeX key | 条目要点 | 核实来源 |
|---|---|---|
| chen2021transunet | TransUNet, arXiv:2102.04306, 2021（无正式期刊出处，按 arXiv 引用） | arxiv.org/abs/2102.04306 |
| cao2022swinunet | Swin-Unet, ECCV 2022 Workshops (LNCS), pp. 205–218, DOI 10.1007/978-3-031-25066-8_9 | Crossref |
| wang2023sammed3d | SAM-Med3D, arXiv:2310.15161（14 位作者顺序逐一核对） | arxiv.org/abs/2310.15161 |
| wu2023medsamadapter | Medical SAM Adapter, arXiv:2304.12620 | arxiv.org/abs/2304.12620 |
| hu2022lora | LoRA, ICLR 2022；arXiv 页无 journal-ref，ICLR 出处由微软官方仓库 README 的 BibTeX 佐证 | raw.githubusercontent.com/microsoft/LoRA/main/README.md |
| dice1945measures | Dice 系数原始文献, Ecology 26(3):297–302, 1945, DOI 10.2307/1932409 | Crossref |
| milletari2016vnet | V-Net（Dice 损失出处），**3DV 2016, pp. 565–571, DOI 10.1109/3DV.2016.79（常被误引为 MICCAI 2016，以 Crossref/IEEE 为准）** | Crossref |
| dosovitskiy2021vit | ViT, ICLR 2021；arXiv Comments 明示 "ICLR camera-ready version" | arxiv.org/abs/2010.11929 |
| zhang2023samed | SAMed（arXiv v2 起更名 "Customized Segment Anything Model for Medical Image Segmentation"）, arXiv:2304.13785 | arxiv.org/abs/2304.13785 |
| long2015fcn | FCN, CVPR 2015, pp. 3431–3440, DOI 10.1109/CVPR.2015.7298965 | Crossref |
| huang2024sam | "Segment anything model for medical images?", Medical Image Analysis 92:103061, 2024（SAM 医学零样本大规模评测，19 位作者逐一核对） | Crossref: 10.1016/j.media.2023.103061 |
| liu2021swin | Swin Transformer, ICCV 2021, pp. 9992–10002, DOI 10.1109/ICCV48922.2021.00986 | Crossref |
| sudre2017generalised | Generalised Dice 损失, DLMIA 2017 (LNCS), pp. 240–248 | Crossref: 10.1007/978-3-319-67558-9_28 |
| cicek2016unet3d | 3D U-Net, MICCAI 2016 (LNCS), pp. 424–432 | Crossref: 10.1007/978-3-319-46723-8_49 |
| zhou2018unetpp | UNet++, DLMIA 2018 (LNCS), pp. 3–11 | Crossref: 10.1007/978-3-030-00889-5_1 |
| shaharabany2023autosam | AutoSAM, arXiv:2306.06370 | arxiv.org/abs/2306.06370 |

## 主动舍弃/未收录（宁缺毋滥）

- 无必引条目被舍弃（missing 为空）。
- 曾考虑但未核实成功而不收录：
  - "Segment Anything Model for Medical Image Analysis? An In-Depth Overview"（未定位到可靠元数据）；
  - MobileSAM、SAM-Med2D、nnSAM、LViT 等（非必需，时间预算内未逐一核实）；
  - "How to build..." 的 Medical Image Analysis 版本（检索确认不存在，真实出处 MELBA 已收录）。

## 写作阶段使用提示

- 全部 key 采用「第一作者姓氏+年份+短词」命名（如 kirillov2023segment），已在 bib 内保持唯一。
- 论文章节映射建议：引言/相关工作引 1–5、11–16、18–25；方法（Dice 定义）引 dice1945measures、sudre2017generalised、milletari2016vnet；数据集引 jha2020kvasir、codella2019skin、jaeger2014two；MedSAM 一节引 ma2024segment、ma2025medsam2、wang2023sammed3d、wu2023medsamadapter、zhang2023samed、shaharabany2023autosam、huang2024sam、koishiyeva2025review、gu2025how。
- 若后续需要英文 abstract 或新增引用，按同样流程先核实再增补本文件清单。
