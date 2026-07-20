# PADC — Part-Aware Descriptor Classifier

Official code for **Part-Aware Descriptor Classifier for Trustworthy Species Detection** (CVPR 2026).

PADC is a hybrid fine-grained classification framework that fuses classical shape descriptors, part-level quality metrics, and deep visual features. A segmentation module localizes anatomical parts; geometric, appearance, and inter-part descriptors are extracted and combined with backbone embeddings through a gated fusion module.

**Paper:** `paper/PADC/main.tex` in the [BeeMachine Codes](https://github.com/kddresearch/BeeMachine) workspace (local path: `Codes/paper/PADC/`).

**Datasets evaluated:** Beemachine (bees), CUB (birds), Fish-Vista (fish).

---

## Repository layout (mapped to the paper)

| Folder | Paper section | Description |
|--------|---------------|-------------|
| `0_data_preprocessing/` | §3 Datasets & Preprocessing | YOLO/COCO mask pipelines, dataset splits (75/15/10), augmentations, species-label generation |
| `1_classification/` | §5 Experiments (baselines) | Backbone-only classifiers (ConvNeXt, EfficientNetV2, DeiT, Swin, ResNet, etc.) — comparison baselines |
| `2_segmentation/` | §4.1 Segmentation (`f_s`) | Part segmentation training (DeepLabV3+, FPN, UNet++, SegFormer, PSPNet, LinkNet, MaNet) |
| `3_padc_experiments/` | §4.6 PADC Variants, §5 Classification | Main PADC experiments per dataset and variant |
| `4_shape_feature_analysis/` | §4.2–4.4 Descriptors | Shape (`φ_s`), appearance/quality (`φ_q`), and inter-part (`φ_c`) feature extraction |
| `dim_reduction/` | §4.5 Dimensionality Reduction | PCA, ICA, TruncatedSVD, NMF, SRP, GRP experiments on descriptor vectors |
| `zero_shot_test/` | §4.6 PADC_Zero | GroundingDINO + SAM 2 zero-shot full-body segmentation and inference |
| `Qualitative_analysis/` | §5.4 Qualitative Analysis | FinerCAM attention-map visualizations |
| `multi_task_detection_model/` | — | Supplementary multi-task detection experiments |
| `others/` | — | Utility notebooks (checkpoint cleanup, etc.) |

### `3_padc_experiments/` structure

Each dataset folder (`Beemachine/`, `CUB/`, `FishVista/`) contains the four PADC variants from the paper:

| Variant | Description |
|---------|-------------|
| `PADC_Part/` | Trained part segmentation + full descriptor set (main model) |
| `PADC_Full/` | Full-body mask only (no part decomposition) |
| `PADC_Red/` | Part descriptors with best dimensionality-reduction method per dataset |
| `PADC_Zero/` | Zero-shot GroundingDINO + SAM 2 segmentation (full-body only) |
| `PADC_Ablation/` | Supplementary ablation notebooks (not reported in main paper tables) |

**Backbones used across experiments:** ConvNeXt-Nano, DeiT, EfficientNetV2-S/M, InceptionNeXt-Tiny, ResNet101, SE-ResNeXt101, Swin.

---

## Workflow

### 1. Environment

Create the conda environment from the parent BeeMachine repo:

```bash
conda env create -f ../beemachine/environment.yaml
conda activate bee_test
```

### 2. Data

| Dataset | Parts | Segmentation subset | Classification subset |
|---------|-------|---------------------|----------------------|
| **Beemachine** | head, thorax, abdomen | 7,716 annotated images (Roboflow) | 34,722 images, 160 species |
| **CUB** | 11 bird parts (body, head, wings, …) | 1,888 images, 70 species | Full CUB-200-2011 |
| **Fish-Vista** | 9 fish parts (fins, head, eye, barbel, …) | 2,427 images | 56,360 images, 1,785 species |

- **Beemachine full dataset:** [HuggingFace Beemachine_2024](https://huggingface.co/datasets/KDDResearch/Beemachine_2024) (5 repos; see BeeMachine README).
- **Beemachine partwhole metadata:** `../beemachine/partwhole_dataset_from_roboflow/`.
- **CUB parts:** [CUB-Part](https://github.com/behzadi-m/cubpart) (Behzadi et al.).
- **Fish-Vista:** [Fish-Vista paper/repo](https://github.com/mehrabianali/Fish-Vista).

### 3. Run experiments (typical order)

1. **Preprocess:** `0_data_preprocessing/yolo_preprocessing_steps_bee/` (Beemachine) or dataset-specific subfolders.
2. **Train segmentation:** `2_segmentation/{Beemachine,CUB,FishVista}/`
3. **Extract descriptors:** `4_shape_feature_analysis/`
4. **Dimensionality reduction (optional, for PADC_Red):** `dim_reduction/`
5. **Train baselines:** `1_classification/{Beemachine,CUB,FishVista}/`
6. **Train PADC variants:** `3_padc_experiments/{dataset}/PADC_{Part,Full,Red,Zero}/`
7. **Zero-shot inference:** `zero_shot_test/`
8. **Qualitative analysis:** `Qualitative_analysis/`

### Example entry points

| Task | Notebook / script |
|------|-------------------|
| Beemachine baseline (Swin) | `1_classification/Beemachine/swinv2_classifier_v5.ipynb` |
| Beemachine PADC_Part (Swin) | `3_padc_experiments/Beemachine/PADC_Part/arch2_PADC_swin.ipynb` |
| CUB shared training logic | `3_padc_experiments/CUB/PADC_Part/PADC_Part_CUB.py` |
| Fish-Vista PADC_Part | `3_padc_experiments/FishVista/PADC_Part/arch2_fish_swin.ipynb` |

---

## Method summary

Given image `I` and label `y`:

1. **Segment** `k` anatomical parts → masks `M_1…M_k` (or full-body mask for PADC_Full / PADC_Zero).
2. **Extract descriptors** `Z_p` = shape (`φ_s`), appearance/quality (`φ_q`), inter-part ratios (`φ_c`).
3. **Encode shape** via lightweight conv encoder `ψ_γ` on aggregated mask → `Z_e`.
4. **Extract visual features** `Z_b` from timm backbone `f_b`.
5. **Gated fusion** `Z_f = Z_b + g ⊙ (Ẑ_e − Z_b)`.
6. **Classify** `Z = [Z_p, Z_f]` through FC head.

---

## Sync with BeeMachine repo

An embedded copy of this code lives at `../beemachine/PADC/`. After editing here, run:

```powershell
..\sync_repos.ps1
```

---

## Citation

```bibtex
@inproceedings{choton2026padc,
  title={Part-Aware Descriptor Classifier for Trustworthy Species Detection},
  author={Choton, Jahid Chowdhury and Campolongo, Elizabeth G and Grijalva, Ivan and Margapuri, Venkat and Spiesman, Brian J and Hsu, William H},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
