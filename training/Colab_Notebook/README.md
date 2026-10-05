# Waste Management AI — 3-Class Object Detection Prototype

## Purpose

This project builds a small, reproducible object-detection pipeline for a waste-management robotics application.

The prototype is designed to be:

* Trainable in Google Colab
* Reproducible
* Transferable to HPC
* Extendable to real competition data
* Deployable on the final robot computer

The overall pipeline is:

```text
Data
→ Labels
→ Train / Validation / Test Split
→ Pretrained YOLO
→ Training
→ Validation
→ Error Analysis
→ Export
→ HPC
```

---

## Prototype Classes

The prototype uses three classes:

| Class ID | Class     | Initial Representation         |
| -------- | --------- | ------------------------------ |
| `0`      | `plastic` | Bottle                         |
| `1`      | `fabric`  | Clothing / fabric-like objects |
| `2`      | `organic` | Watermelon                     |

> **Important:** These are proxy classes for the initial prototype. A bottle does not automatically represent all plastic waste, and clothing does not represent all fabric waste. The competition model should eventually be trained using actual waste materials and objects from the game field.

---

## Engineering Targets

The initial engineering targets are:

* Minimum validation **mAP50 = 0.80**
* Preferred validation **mAP50 ≥ 0.90**
* Target inference speed **≥ 20 FPS** on the final robot computer
* Camera target: **30 FPS**
* Deployment: **onboard robot computer**
* Exact deployment hardware/GPU: to be confirmed

The model should also be tested against difficult conditions, including:

* Poor lighting
* Shadows
* Partially hidden objects
* Touching or overlapping objects
* Small or far-away objects
* Unusual object orientations
* Motion blur
* Dirty or crushed objects
* Confusing backgrounds
* False detections on field boards or the floor

---

# 1. Environment Setup

The prototype is designed to run in Google Colab.

Install the required packages:

```python
!pip -q install -U ultralytics fiftyone
```

The main packages used by the pipeline are:

```text
ultralytics
fiftyone
pyyaml
numpy
pillow
matplotlib
```

---

# 2. Project Configuration

The project uses the following directories:

```text
/content/waste_ai/
├── raw/
├── dataset/
├── exports/
└── runs/
```

The main paths are:

```python
PROJECT = Path("/content/waste_ai")
RAW_DIR = PROJECT / "raw"
DATASET_DIR = PROJECT / "dataset"
EXPORT_DIR = PROJECT / "exports"
```

The prototype classes are defined as:

```python
CLASSES = ["plastic", "fabric", "organic"]
```

with:

```text
0 = plastic
1 = fabric
2 = organic
```

The random seed is:

```python
SEED = 42
```

---

# 3. Dataset

The prototype uses a small subset of **Open Images V7** through FiftyOne.

The initial dataset uses:

```text
Bottle
Watermelon
Clothing
```

with the following mapping:

```text
Bottle      → plastic
Watermelon  → organic
Clothing    → fabric
```

The prototype limits the number of downloaded samples per class so that the complete pipeline can be tested quickly in Colab.

---

# 4. Real Fabric Dataset

The `fabric` class currently uses clothing as a temporary proxy.

Real fabric images should eventually be added under:

```text
/content/waste_ai/raw/fabric_real/
```

These images should represent the actual fabric waste/material expected in the competition.

If the images do not already have bounding-box annotations, they should be annotated before being added to the training dataset.

Suitable annotation tools can include:

* CVAT
* Roboflow

The final competition dataset should prioritize photographs of the actual objects and materials rather than relying on proxy classes.

---

# 5. YOLO Dataset Format

The dataset is converted into YOLO detection format.

The directory structure is:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image has a corresponding `.txt` label file.

YOLO annotations use:

```text
class_id x_center y_center width height
```

All coordinates are normalized between `0` and `1`.

For example:

```text
0 0.523 0.471 0.312 0.428
```

where:

```text
0 = plastic
```

---

# 6. Dataset Export

The dataset conversion process:

1. Reads the source dataset.
2. Extracts the detections.
3. Maps the source class names to the project classes.
4. Converts bounding boxes into YOLO format.
5. Saves the images.
6. Saves the corresponding `.txt` annotation files.

The final dataset is structured into:

```text
train
val
test
```

for both images and labels.

---

# 7. Train / Validation / Test Split

The prototype uses the following split:

```text
70% → Training
20% → Validation
10% → Test
```

The three sets have different purposes:

### Training

Used to train the model.

### Validation

Used during model development to evaluate performance and guide training decisions.

### Test

Used as the final evaluation set.

The test set should not be repeatedly used during model tuning.

### Competition Dataset

For real competition data, the split should be based on:

```text
Scene
Video
Session
```

rather than randomly splitting adjacent video frames.

This helps prevent nearly identical frames from appearing in both training and test data.

---

# 8. Class Balance

Before training, the number of annotations in each class should be checked.

The three classes are:

```text
plastic
fabric
organic
```

Class balance should be checked across the dataset to identify situations where one class has significantly fewer examples than another.

---

# 9. YOLO Data Configuration

The dataset uses a `data.yaml` file.

Example:

```yaml
path: /content/waste_ai/dataset

train: images/train
val: images/val
test: images/test

names:
  0: plastic
  1: fabric
  2: organic
```

This file should be included when the trained project is transferred to HPC.

---

# 10. Dataset Visual Sanity Check

Before training, the dataset should be visually inspected.

Check that:

* Bounding boxes are positioned correctly.
* Class IDs are correct.
* Images correspond to their labels.
* Objects are actually visible.
* Images are not corrupted.
* The dataset contains realistic examples.

The visual inspection step is important.

> If the bounding boxes or class IDs are wrong, stop and fix the dataset before training.

---

# 11. Training

The prototype uses a pretrained YOLO model:

```text
yolo26n.pt
```

The initial training configuration is:

```text
Epochs:     30
Image size: 640
Batch:      -1
Patience:   10
```

Training output is stored under:

```text
/content/waste_ai/runs/
```

The training run is named:

```text
prototype
```

The best trained model is:

```text
/content/waste_ai/runs/prototype/weights/best.pt
```

The initial training configuration is intended to verify the complete pipeline.

It is not necessarily the final configuration for competition training.

---

# 12. Validation

After training, the best model is evaluated on the validation dataset.

The main metrics are:

* mAP50
* mAP50-95
* Precision
* Recall
* Per-class performance

The validation results should be recorded for every significant experiment.

Particular attention should be paid to per-class performance because a strong overall metric can hide poor performance on an individual class.

---

# 13. Test Set

The test set is treated as the final exam for the model.

The model should be evaluated on the test set after the main development and tuning process.

Record:

```text
Test mAP50
Test mAP50-95
Test Precision
Test Recall
Per-class performance
```

Avoid repeatedly tuning the model against the test set because this can cause the test set to become part of the development process.

---

# 14. Prediction Inspection

Metrics are not enough.

The model should also be run on test images so the actual predictions can be inspected.

Example:

```python
results = model.predict(
    source=test_images,
    conf=0.25,
    save=True
)
```

Inspect predictions for:

* Missed objects
* Incorrect classes
* Incorrect bounding boxes
* False positives
* Small-object failures
* Overlapping-object failures
* Background confusion

This is the main error-analysis stage of the pipeline.

---

# 15. Inference Speed

Inference speed should be measured during development.

However, Colab inference speed should only be treated as a rough benchmark.

The final FPS must be measured on the actual robot computer.

The target is:

```text
≥ 20 FPS
```

with a camera target of:

```text
30 FPS
```

The final deployment benchmark should consider the complete perception pipeline rather than only the raw model inference time.

---

# 16. Export

The main model artifact is:

```text
best.pt
```

The following should be preserved:

```text
best.pt
data.yaml
dataset
training results
notebook
requirements.txt
README.md
```

The export directory is:

```text
/content/waste_ai/exports/
```

A requirements file is generated containing:

```text
ultralytics
fiftyone
pyyaml
numpy
pillow
matplotlib
```

---

# 17. HPC Package

The trained prototype can be packaged for transfer to the HPC environment.

The package should contain the important components required to reproduce the experiment:

```text
waste_ai/
├── dataset/
├── exports/
├── runs/
├── data.yaml
├── requirements.txt
├── README.md
└── notebook
```

The purpose of the HPC transfer is to allow larger-scale training using the available HPC GPU resources.

---

# 18. HPC Workflow

The planned workflow is:

```text
Prototype in Colab
        ↓
Export Dataset + Configuration
        ↓
Transfer to HPC
        ↓
Create Python Environment
        ↓
Use HPC GPU
        ↓
Run Training
        ↓
Save Checkpoints
        ↓
Evaluate Model
        ↓
Save best.pt
```

The same dataset configuration and experiment information should be preserved when moving between environments.

---

# 19. Development Roadmap

## V1 — Prototype

The current prototype consists of:

* 3 classes
* Public proxy data
* Pretrained YOLO
* Google Colab training
* Validation
* Test evaluation
* Model export

---

## V2 — Real Objects

Replace the proxy data with photographs of the actual competition objects:

```text
Plastic waste
Fabric waste
Organic waste
```

The dataset should represent the actual objects that the robot will encounter.

---

## V3 — Competition Environment

Collect data using the actual robot camera and the actual game-field environment.

The dataset should include:

* Field boards
* Floor
* Lighting variations
* Shadows
* Different object distances
* Different object placements
* Clutter
* Realistic backgrounds

---

## V4 — Robotics Integration

The intended robotics pipeline is:

```text
Camera
   ↓
YOLO Detection
   ↓
Object Position
   ↓
ROS 2
   ↓
Navigation / Manipulation
```

---

## V5 — HPC Training

Move the training pipeline to the HPC environment.

The workflow becomes:

```text
Dataset
   ↓
Python Environment
   ↓
HPC GPU
   ↓
Batch Training
   ↓
Checkpoints
   ↓
best.pt
```

---

## V6 — Deployment

Benchmark the trained model on the actual robot computer.

Depending on the final hardware and performance requirements, possible deployment optimizations include:

* Smaller model
* Lower image resolution
* ONNX
* TensorRT
* Other optimized formats

---

## V7 — Robustness

The model should be improved iteratively using:

```text
Collect
   ↓
Label
   ↓
Train
   ↓
Validate
   ↓
Inspect Errors
   ↓
Collect More
   ↓
Retrain
```

The focus is on collecting difficult examples from the real deployment environment.

---

# 20. Experiment Record

Every significant experiment should record:

```text
Dataset version
Number of images
Images per class
Model
Epochs
Image size
Batch size
GPU
mAP50
mAP50-95
Precision
Recall
Per-class performance
Inference FPS
Main failure cases
What changed
```

This allows different experiments to be compared and reproduced later.

---

# Important Notes

This is a **prototype training pipeline**.

The current public dataset is intended to validate the machine-learning workflow. It should not be treated as the final competition dataset.

The next major step is to collect and label real images of the waste materials using the actual deployment camera and environment.

The final model should therefore be evaluated using:

```text
Actual waste objects
+
Actual robot camera
+
Actual game field
+
Actual lighting
+
Actual deployment hardware
```

rather than relying only on the initial Colab prototype results.
