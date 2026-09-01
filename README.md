# Waste Classification Dataset

This dataset is prepared for training a lightweight classifier (e.g., MobileNetV2) for a waste-sorting robot to distinguish between organic, metal, and plastic materials.

## Data Sources
- **Sources Used:** Kaggle 12-class Garbage Classification dataset (supplemented with TrashNet data).
- *Class Mapping:* The Kaggle `biological` class was mapped to `organic`.

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