# 医学影像分割：从 SAM 到 MedSAM

本项目研究医学影像分割中提示形式和标注规模的影响：比较 MedSAM 零样本推理、U-Net 少样本训练及跨数据集泛化。论文与完整方法见 [paper/main.pdf](paper/main.pdf)，实验记录见 [results/](results/)。

## 方法与结果

在 Kvasir-SEG 息肉、ISIC2018 皮肤病变和 Montgomery 肺野数据上，使用逐样本宏平均 Dice 评估。MedSAM 使用由真值掩码得到的外接框、扰动框及由前景点构造的伪框；后两种点实验并非原生点提示。U-Net 在前两个数据集分别用 20、100 和完整训练池（720 / 1868 张）训练，并评估跨数据集表现。

| 测试集 | MedSAM 真值框 Dice | U-Net 完整训练池 Dice |
| --- | ---: | ---: |
| Kvasir-SEG（200 张） | 0.930684 | 0.570244 |
| ISIC2018（500 张） | 0.929824 | 0.829606 |
| Montgomery（28 张） | 0.976842 | 未训练 |

详见 [metrics.csv](results/metrics.csv) 和论文。真值框依赖标注，结果不能直接等同于临床真实使用表现；点实验使用伪框、不是原生点编码；数据集规模和模态有限，U-Net 的比较也受训练预算与实现影响，不能推广为所有模型和场景的结论。

## 内容与复现

- `paper/`：论文源码、参考文献和 PDF；`main.pdf` 是相同的最新论文。`main(1).pdf` 保留此前排版版本。
- `results/`：逐样本指标、汇总和部分预测掩码；`figures/`：论文图表。
- `data/`：数据集划分与统计。原始影像、标注和模型权重不在仓库中；数据来源和处理见 [data/README.md](data/README.md)，使用前请遵守各数据集及权重的原始许可和使用条件。
- `scripts/`：数据准备、推理、训练及图表脚本。需要 Python、PyTorch、NumPy、Pillow、Matplotlib 和 `segment-anything`；需按平台配置 PyTorch 环境。完整的依赖版本锁定文件未提供。

论文需要 XeLaTeX、BibTeX 和相应中文字体及宏包；在 `paper/` 下运行：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

实验复现前，按 `data/README.md` 准备三个数据集，并将 [MedSAM 官方权重](https://zenodo.org/records/10689643) 放到 `checkpoints/medsam_vit_b.pth`。然后可运行：

```sh
python scripts/medsam_infer.py
python scripts/unet.py --dataset Kvasir-SEG --tier 20
python scripts/unet.py --dataset ISIC2018 --tier full
python scripts/cross_dataset.py
```

这些命令会进行推理或训练，并可能更新 `results/`；已有结果会被部分脚本用于跳过已处理样本。完整 U-Net 对比需分别运行两个数据集的 `20`、`100`、`full` 档位；跨数据集评估依赖对应训练权重。复现条件、数据准备和训练耗时依硬件与数据获取情况而异。
