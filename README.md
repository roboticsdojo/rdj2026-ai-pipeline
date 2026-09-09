# Waste Classification Dataset

This dataset is prepared for training a lightweight classifier (e.g., MobileNetV2) for a waste-sorting robot to distinguish between organic, metal, and plastic materials.

## Data Sources
- **Sources Used:** Kaggle 12-class Garbage Classification dataset (supplemented with TrashNet data).
- *Class Mapping:* The Kaggle `biological` class was mapped to `organic`.

## Dataset Setup
To recreate the raw dataset images locally, follow these steps:
1. Download the "Garbage Classification (12 classes)" dataset from Kaggle and save the downloaded `archive.zip` in your `Downloads` folder (or update the script path).
2. Run the `setup_datasets.py` script, which will automatically clone TrashNet, extract both datasets, and correctly organize them into the `dataset/raw/` directory.

```bash
python3 setup_datasets.py
```

After setup, you can run `python3 prepare_dataset.py` to build the final augmented dataset.

## Final Image Counts (per class/split)
The `dataset/final` folder contains the following splits (70/20/10 ratio):

| Class   | Train (Real) | Train (Augmented x3) | Val (Real) | Test (Real) |
|---------|--------------|----------------------|------------|-------------|
| Organic | 689          | 2756                 | 197        | 99          |
| Metal   | 538          | 2152                 | 153        | 78          |
| Plastic | 608          | 2432                 | 173        | 88          |

## Caveats & Known Issues
- All classes are now populated and reasonably well-balanced (ranging between 2100 and 2700 images per class in the augmented training set).
- Duplicate checking effectively filtered duplicate images between the combined public datasets.