# Findings & Experiments

This is the story of how DirhamVision was built — the dataset, the experiments that shaped it, what worked, what didn't, and what it would take to turn this proof of concept into something production-ready.

The one-line takeaway: **data quantity was the bottleneck at every stage.** Model size, augmentation, and training length all mattered far less than what was in the dataset and how it was labeled.

---

## Contents

1. [The dataset](#1-the-dataset)
2. [Experiment 1 — labeling the bill](#2-experiment-1--labeling-the-bill)
3. [Experiment 2 — nano vs small](#3-experiment-2--nano-vs-small)
4. [Experiment 3 — augmentation](#4-experiment-3--augmentation)
5. [The final model](#5-the-final-model)
6. [Known limitations](#6-known-limitations)
7. [Why no separate test set](#7-why-no-separate-test-set)
8. [What production would require](#8-what-production-would-require)

---

## 1. The dataset

No public dataset with both AED coins and banknotes existed, so I built one from scratch — shot on a phone, labeled by hand, and rebuilt several times as problems showed up.

The first version (~800 images) looked fine on paper but fell apart in real-world video. The cause was a distribution shift: too many hard, mixed, and occluded images and not enough clean, solo, straightforward ones. So I rebuilt it with better conditions and a healthy mix of solo and multi-coin shots.

**Final dataset (709 images, 4 classes):**

| Class | Instances | Solo images | Mixed images |
|---|---|---|---|
| 25 fils | 242 | 146 | 93 |
| 50 fils | 265 | 160 | 97 |
| 1 AED coin | 350 | 141 | 97 |
| 10 AED bill | 217 | 143 | 51 |

103 images had more than one class in frame. I also added 14 background images (no coins or bills at all) as negative samples — about 2% of the dataset, which is the amount Ultralytics recommends. These teach the model what *not* to fire on.

Building this myself taught me more about object-detection data than any tutorial — how distribution shift wrecks a model, why class balance matters, and how much clean examples carry the whole thing.

---

## 2. Experiment 1 — labeling the bill

Before settling on how to label the 10 AED note, I tried something specific: instead of boxing the whole bill, label only the **"TEN DIRHAMS" text** — in both Arabic and English — since that text is high-contrast and consistent. The idea was that a sharp, reliable feature might be more precise than a big box around the whole note.

It half-worked, and that's the interesting part.

| Model | Bill mAP@50 | Confusion-matrix diagonal |
|---|---|---|
| yolo26s (one-feature-only) | 0.823 | 0.59 — *41% of bills missed* |
| yolo26n (one-feature-only) | 0.809 | 0.76 — *24% of bills missed* |

When the model saw the words, it was confident and correct. But two problems made it unusable for a real application:

- **High miss rate.** When neither text block was clearly and fully visible (folded, angled, partially covered), the model often missed the bill entirely, and by nature of design, you can't find a feature to reliably target that will be visible on all sides in all conditions.
- **Double counting.** A workaround I thought of is to target a different feature when the text wasn't visible, but this resulted in double counting when the model could see them both, and non-consistency across a class largely harms performance.

Then I relabeled every bill with a **whole-bill box** and retrained on the exact same images:

| Model | Bill mAP@50 | Confusion-matrix diagonal |
|---|---|---|
| yolo26s (whole-bill) | 0.897 | 0.89 |
| yolo26n (whole-bill) | 0.877 | 0.93 |

Whole-bill won clearly. The feature-only idea was worth testing — it told me you *can* squeeze out high precision by targeting a feature — but it can't be relied on for this use case. Even with more data, the double-counting and the missed folded bills don't go away. They're baked into the approach.

---

## 3. Experiment 2 — nano vs small

The goal was always the **nano** model. Assistive tech has to run on a phone, not a server. But I trained the **small** model alongside it at every step as a sanity check — was nano costing me real accuracy?

It wasn't.

| Model | Params | Inference | mAP@50 (whole-bill, 50ep) |
|---|---|---|---|
| yolo26n | 2.4M | ~4ms | 0.952 |
| yolo26s | 9.5M | ~7ms | 0.964 |

The small model was ahead by about a single mAP point — but it ran slower and, on the confusion matrices, hallucinated *more* (firing on empty background more often). On a dataset this size, the extra capacity didn't help — larger models have more parameters to update, which increases overfitting risk when training data is limited, and that showed up in practice as higher false-positive rates on background rather than better generalization. For an edge-deployment project, nano was the right call on both the goal and the numbers.

---

## 4. Experiment 3 — augmentation

The augmentation settings weren't copied from a recipe — each one came from a failure I actually saw in video. I initially attempted aggressive augmentation: an aggressive run on the small model made things *worse*, pushing the background→1 AED false-positive rate up to 0.56 from 0.40.

After some trial and error here are the final settings that maxamized the results, and the reason for each:

| Setting | Value | Default | Why |
|---|---|---|---|
| `mosaic` | 0.0 | 1.0 | Recommended off for small datasets; it was creating unrealistic chopped-up coin images |
| `degrees` | 180 | 0.0 | Video showed coins getting missed when upside-down — full rotation fixed it |
| `scale` | 0.2 | 0.5 | A little distance variation helps; too much shrinks coins into mush |
| `erasing` | 0.0 | 0.4 | Randomly blacking out chunks of an already-tiny coin does more harm than good |

Everything else stayed at YOLO's defaults. The rule I followed: only change a default when there's a specific problem it fixes.

---

## 5. The final model

**YOLO26n · 140 epochs · tuned augmentation · 2% background images.** Training stopped itself early at epoch 116.

| Metric | Value |
|---|---|
| mAP@50 | **0.982** |
| mAP@50-95 | 0.902 |
| Inference | 3.9ms |
| Parameters | 2.4M |

**Per class:**

| Class | mAP@50 | Recall |
|---|---|---|
| 10 AED bill | 0.994 | 0.966 |
| 1 AED coin | 0.985 | 0.939 |
| 50 fils | 0.985 | 0.951 |
| 25 fils | 0.962 | 0.852 |

The whole-bill relabeling turned the banknote from the weakest class into the strongest. Error analysis on the validation set found 13 problem images out of 104 — only 2 false positives and 3 wrong-class calls. The rest were misses at awkward angles, not dangerous mix-ups.

> **Placeholder:** `assets/confusion_matrix.png` — final model confusion matrix

In real-world video the final model beat every earlier version: the 1 AED / 25 fils flicker mostly vanished, bills stopped getting double-counted, and the 50 fils held steady.

---

## 6. Known limitations

This is a proof of concept, and it has real weak spots:

**25 fils vs 1 AED — the hardest problem.** These two coins are both grey circles with no real distinguishing feature. The only thing that actually separates them is *size* — and here's the catch: an object detector doesn't measure absolute size. Scale augmentation deliberately trains it to recognize a coin whether it's near or far. So the model can only tell these two apart using *relative* size, which it can only see when both coins are in the same frame to compare. That's why 25 fils is the weakest class, and why the fix is data, not a smarter model: it needs far more mixed images where 25 fils and 1 AED appear together, at different distances and lighting.

**Double counting on coins.** A single coin occasionally gets detected twice. Raising the confidence threshold fixes it but causes more misses. The real fix is more data.

**Stacked bills.** Two overlapping notes sometimes read as one. Worth noting: this problem *didn't* happen with the text-only labeling from Experiment 1 — but that approach had bigger problems elsewhere. With whole-bill labeling, the clean fix is to deliberately feed the model lots of stacked-bill examples so it learns that exact scene. Train for the hard case directly.

**Distance and bad lighting.** Coins far away, in dim light, or on a low quality webcam drop off fast. But this is partly the nature of the problem — coins are grey and featureless, and even a *person* could struggle to identify a coin from a distance. The dataset did include hard conditions, but it doesn't have enough distance and lighting variety to be robust everywhere.

Every one of these comes back to dataset size and coverage — none to the architecture or the training.

---

## 7. Why no separate test set

I deliberately didn't hold out a separate test set. With ~700 images, a three-way split would have left around 100 images each for validation and testing — both drawn from the same pile, giving me one extra number that wouldn't really tell me anything new.

The real test was **video**. I shot footage completely separate from the training data, in real conditions, in a different format (live video vs still photos), and never tuned anything based on what I saw in it. That's a tough test and it caught problems (flicker, double-counting) that the validation numbers alone never showed.

---

## 8. What production would require

The pipeline, the labeling approach, the augmentation, and the model all work. What's missing is data, and specifically the *right kind* of data.

The single most important lesson for anyone extending this: **the look-alike coins need to appear together, constantly.** The 25 fils and 1 AED problem is only solvable when the model repeatedly sees them side by side — and not just with each other, but mixed in with bills and other coins too, at all sorts of distances and lighting. A coin that has no unique shape can only be learned through relative size, and relative size only exists in mixed frames. So those two coins should show up in the large majority of your multi-object images.

**Rough data targets:**

| Scope | Estimated images |
|---|---|
| Current 4 classes, production-quality | ~1,500–2,000 per class, heavy on mixed scenes |
| The two look-alike coins (25 fils, 1 AED) | Over-represented — present in most mixed images |
| Full AED set (16 denominations) | ~15,000–25,000 total, balanced and varied |

More-used denominations should be well-represented — especially visually similar coins where size is the only distinguishing factor, as discussed earlier. Less common denominations (the larger notes, the smaller fils coins) can be under-represented, but not so severely that the model can't recognize them at all or — worse — mistakes them for something else.

If you've read this far, thank you. I hope the experiments and findings here are useful to anyone working on currency recognition or similar small-dataset detection problems. I'm always open to discussing the project, the methodology, or anything in between — feel free to reach out.

---

*Built by Ahmed Mohamed — [GitHub](https://github.com/v7wed) · [LinkedIn](https://www.linkedin.com/in/v7wed/)*