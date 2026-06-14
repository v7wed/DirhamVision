<div>

# DirhamVision

Real-time UAE currency detection and counting — built on YOLO26n.

> **💡 Proof of Concept** — DirhamVision explores what's possible with a small, custom-built AED currency dataset. It covers 3 coins (25 fils, 50 fils, 1 AED) and 1 banknote (10 AED), trained on ~700 images. It performs well under controlled conditions, but it isn't reliable in hard conditions and doesn't cover every denomination yet — poor lighting, distance, and heavy occlusion all degrade results. More data is the path to production quality. If you're curious about exactly what that takes, along with the experiments and findings behind the model, see [FINDINGS.md](FINDINGS.md).

</div>

---
## What it does

<table>
<tr>
<td valign="top">

DirhamVision detects dirham denominations from a live camera, video file, or image and displays a running total in real time. The target use case is assistive technology for visually impaired users — knowing how much cash you're holding without needing to ask someone.

**Supported denominations**

| Class | Value |
|---|---|
| 25 fils coin | 0.25 AED |
| 50 fils coin | 0.50 AED |
| 1 AED coin | 1.00 AED |
| 10 AED banknote | 10.00 AED |

**Model** — YOLO26n (nano), 2.4M parameters, ~4ms inference, 98.2% mAP@50 on validation set.

</td>
<td valign="top" width="200">
<img src="assets/demo.gif" width="180"/>
</td>
</tr>
</table>

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/v7wed/DirhamVision.git
cd DirhamVision
pip install -r requirements.txt
```

### 2. Run

```bash
python scripts/detect.py
```

**Run on a video or image**

```bash
# Video file
python scripts/detect.py --source path/to/video.mp4

# Image
python scripts/detect.py --source path/to/image.jpg
```

**Save output to file instead of displaying**

```bash
python scripts/detect.py --source path/to/video.mp4 --save
# Output saved to outputs/
```

**All flags**

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | `0` = webcam, or path to video/image |
| `--weights` | `weights/dirhamvision_nano.pt` | Path to model weights |
| `--conf` | `0.35` | Confidence threshold |
| `--save` | `False` | Save output to `outputs/` instead of displaying |

---

## Model weights

Both weights are included in the `weights/` folder.

| File | Description | mAP@50 |
|---|---|---|
| `dirhamvision_nano.pt` | **Shipping model** — YOLO26n, 140 epochs, tuned augmentation | 98.2% |
| `dirhamvision_nano_baseline.pt` | Baseline — YOLO26n, 50 epochs, no augmentation | ~96% |

> The default `--weights` flag points to `dirhamvision_nano.pt`.

---

## Train further on more data

The trained weights in this repo are a strong starting point — they already understand AED coin shapes, the 10 AED banknote, and real-world lighting variation. Fine-tuning from `dirhamvision_nano.pt` rather than a generic pretrained model will get you better results faster, especially for adding new denominations or improving performance in specific conditions.

**Method A — Local**

1. Place your dataset in the `data/` folder
2. Update `data/data.yaml` with the correct path:

```yaml
path: ./data                           # ← relative to repo root, or use absolute path
train: train/images
val: val/images
nc: 4
names:
  - 10_aed_bil
  - 1_aed_coin
  - 25_fils
  - 50_fils
```

3. Open `notebooks/DirhamVision.ipynb` locally and run from **Section 3 — Training**, using `weights/dirhamvision_nano.pt` as the starting point. Sections 1 and 2 are Colab/Drive specific and can be skipped.

**Method B — Google Colab**

1. Upload your dataset as a `.zip` to Google Drive
2. Open `notebooks/DirhamVision.ipynb` in Colab
3. In **Section 2**, update `DATASET_ZIP` to point to your file
4. Run all cells — the notebook handles splitting, validation, and training end to end

---

## Project structure

```
DirhamVision/
├── notebooks/
│   └── DirhamVision.ipynb          # Full training pipeline (Colab or local)
├── scripts/
│   └── detect.py                   # Inference — webcam, video, image
├── weights/
│   ├── dirhamvision_nano.pt        # Shipping model
│   └── dirhamvision_nano_baseline_s.pt
├── assets/
│   ├── demo.gif
│   └── confusion_matrix.png
├── data/
│   └── data.yaml                   # Dataset config — update path before training
├── FINDINGS.md                     # Dataset decisions, experiments, limitations
├── requirements.txt
└── .gitignore
```

---

## Findings and experiments

No suitable public dataset that included both AED coins and bills was found, so everything here was built from scratch across multiple iterations. The full write-up covers:

- How dataset quantity was the ceiling at every stage — not model size or architecture
- The labeling strategy experiment (feature-anchored vs whole-object bounding boxes) and what the results showed
- The 25 fils / 1 AED confusion problem, why it happens, and what the data fix looks like
- Known limitations and what a production-quality version would realistically need

→ [Read FINDINGS.md](FINDINGS.md)

---

## Author

Built by Ahmed Mohamed

[![GitHub](https://img.shields.io/badge/GitHub-v7wed-181717?logo=github)](https://github.com/v7wed)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-v7wed-0A66C2?logo=linkedin)](https://www.linkedin.com/in/v7wed/)

---

## License

MIT — weights, dataset insights, and code are free to use in your own projects. See [LICENSE](https://github.com/v7wed/DirhamVision/blob/main/LICENSE.MD) for details.