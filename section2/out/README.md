# Section 2: Hippocampus Segmentation with a 2D U-Net

This section trains a **2D U-Net** to segment the hippocampus (anterior and posterior parts) in cropped brain MRI, and evaluates it on **whole 3D volumes of unseen patients**, the same way the model is used in deployment.

Input: the 260 curated image/label pairs from Section 1 (`../section1/out`).

## Results (52 held-out test patients)

| Metric | Result |
|---|---|
| **Mean Dice** | **0.896** |
| **Mean Jaccard (IoU)** | **0.813** |
| Mean sensitivity | 0.875 |
| Mean specificity | 0.998 |
| Training | 10 epochs, 37 min on Apple Silicon GPU (MPS) |

Per-patient scores and the full configuration are in [`out/<run>/results.json`](out/).

## Training curves and predictions

| Loss (train and validation) |
|---|
| ![Loss](out/tensorboard_loss.png) |

| Step 0 | Step 500 | Step 999 |
|---|---|---|
| ![step 0](out/tensorboard_step0.png) | ![step 500](out/tensorboard_step500.png) | ![step 999](out/tensorboard_step999.png) |

Each screenshot shows the ground-truth mask, the predicted mask and the probability map. Train and validation loss stay close together, so there's no meaningful overfitting. Most of the learning happens in the first 2–3 epochs.

## Method

| Step | Choice | Why |
|---|---|---|
| Loading | `nibabel` instead of the starter's `medpy` | one NIfTI library across the project, fewer dependencies |
| Normalization | each volume divided by its max → [0, 1] | the dataset mixes uint8 and float32 images with very different ranges |
| Reshaping | zero-pad slices to 64×64 (top-left) | keeps true voxel size, so volumes in mm³ stay correct (resizing would distort them) |
| Split | by **patient**, 60/20/20, seed 42 → 156 / 52 / 52 | no slices of one patient in two sets → no data leakage; reproducible |
| Model | recursive 2D U-Net (DKFZ), 3 classes, ~31M parameters | standard architecture for medical segmentation |
| Training | cross-entropy, Adam (lr 2e-4), batch 8, 10 epochs | starter defaults, enough to reach the target Dice |
| Evaluation | predict full volumes slice by slice, then argmax | matches how the model runs on a real scan |
| Metrics | Dice, Jaccard, sensitivity, specificity on **binarized** masks | labels contain 0/1/2; a naive `sum(a*b)` would count class 2 wrongly |

## Key finding: the model under-segments

Sensitivity (0.875) is lower than precision (about 0.92, derived from Dice and sensitivity). The model rarely paints background, but misses about 12% of the true hippocampus voxels, mostly at thin edges. The weakest test cases all show this pattern:

| | Patient | Dice | Sensitivity | Specificity |
|---|---|---|---|---|
| Best | `hippocampus_053` | 0.930 | 0.928 | 0.998 |
| Best | `hippocampus_227` | 0.929 | 0.908 | 0.999 |
| Worst | `hippocampus_070` | 0.826 | 0.739 | 0.999 |
| Worst | `hippocampus_221` | 0.800 | 0.770 | 0.997 |

**Clinical implication:** reported hippocampal volumes will tend to be slightly **too small**. A small hippocampus is the warning sign in Alzheimer's disease, so this bias has to be measured and corrected before clinical use.

Specificity looks almost perfect (0.998) because background, including the zero padding, makes up about 98% of each volume. It says little on its own.

## How to run

From this folder (`section2/`), with the environment from the main README:

```bash
cd src
python run_ml_pipeline.py          # loads data, splits, trains, tests, writes out/<run>/results.json
tensorboard --logdir runs          # in a second terminal: training curves and images
```

In `run_ml_pipeline.py` → `Config`:

- `smoke_test = True` runs the whole pipeline on 20 patients for 1 epoch (about 30 s), to catch bugs before a full run.
- `smoke_test = False` is the full training (about 37 min on an M-series MacBook).

The device is picked automatically: CUDA, then Apple MPS, then CPU.

Model weights (`model.pth`, ~124 MB) are not tracked in git; see the repository's **Releases** page.

## Code structure

```
src/
├── run_ml_pipeline.py              # entry point: config, load, split, train, test, save results
├── data_prep/
│   ├── HippocampusDatasetLoader.py # load NIfTI, normalize, pad to 64x64
│   └── SlicesDataset.py            # serves single 2D slices to the DataLoader
├── experiments/UNetExperiment.py   # training / validation loop, TensorBoard, testing
├── inference/UNetInferenceAgent.py # predicts a whole 3D volume slice by slice
├── networks/RecursiveUNet.py       # U-Net architecture (DKFZ)
└── utils/
    ├── utils.py                    # padding (med_reshape), TensorBoard helpers
    └── volume_stats.py             # Dice, Jaccard, sensitivity, specificity
```

## Limitations and next steps

- **Under-segmentation:** try a Dice or combined Dice + cross-entropy loss, which handles the small foreground better.
- **Checkpointing:** keep the model with the lowest validation loss and add early stopping. Validation loss plateaued after about 4 epochs.
- **Per-class metrics:** scores currently merge anterior and posterior; class-wise Dice would reveal confusion between the two parts.
- **2D only:** a 2.5D or 3D U-Net could use context across slices.
- **Single data source** without demographic or diagnostic metadata: generalization to other scanners and populations is untested.
