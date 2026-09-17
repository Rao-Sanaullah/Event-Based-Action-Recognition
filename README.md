# Event-Based Action Recognition

**A manually implemented, framework-free convolutional Spiking Neural Network (SNN) for object and manipulation-action recognition from event-camera data.**

🔗 **Project page (docs, demos, results):** [https://rao-sanaullah.github.io/Event-Based-Action-Recognition/](https://rao-sanaullah.github.io/Event-Based-Action-Recognition/)

---

## Overview

This repository provides a from-scratch, dependency-free convolutional SNN — built entirely from leaky integrate-and-fire (LIF) neurons and a surrogate-gradient training method, with no reliance on existing SNN libraries (snnTorch, SpikingJelly, etc.). It targets two event-camera classification tasks:

- **Object classification** (5-class)
- **Fine-grained object-action classification** (25-class)

The codebase documents and resolves two previously undocumented training failure modes for manually implemented SNNs, evaluates three input event representations (combined-polarity count, binary event-presence, and polarity-split), and provides a rigorously validated hierarchical object-then-action meta-ensemble.

### Key results

| Task | Best result | vs. published baseline |
|---|---|---|
| Object classification (5-class) | **98.73%** (polarity-split ensemble) | 93.33% (MobileNetV2-3D) |
| Fine-grained object-action (25-class) | **92.99%** (polarity-split ensemble) | 86.80% (MobileNet Transformer) |
| Fine-grained, hierarchical meta-ensemble (combined-polarity) | 88.13% ± 0.96% (10-split validated) | — |

All improvements over the combined-polarity baseline are confirmed statistically robust via paired bootstrap resampling (2000 resamples; 95% CI excludes zero for both tasks).

---

## Features

- **From-scratch SNN architecture**: two convolutional blocks + fully-connected layer, all with LIF neurons and a fast-sigmoid surrogate gradient
- **Event-to-spike-train conversion** for three input encodings (combined-polarity, binary presence, polarity-split)
- **Training pipeline** with focal loss, spike-dropout and time-shift augmentation, and macro-accuracy-based checkpoint selection (fixes a documented failure mode where loss-based selection favors under-trained models)
- **8-model ensembling** for all reported results
- **Hierarchical object-then-action decomposition** with a probabilistic meta-ensemble, validated via repeated held-out splits
- **Statistical validation tooling**: paired bootstrap confidence intervals, single-sample deployment-realism testing
- **Interactive demonstration tool**: replays a converted sample step-by-step, showing live ensemble confidence alongside incoming spike activity

---

## Repository Structure
