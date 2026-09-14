import os
from ultralytics import YOLO

def main():
    # Load a pretrained YOLO classification model
    # (Using the classification variant '-cls' since this dataset is for image classification)
    model = YOLO('yolov8n-cls.pt')

    # Get absolute path to the prepared dataset
    # This script is in training/scripts/, so dataset is at ../../dataset/final
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/final'))

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure you have run setup_datasets.py and prepare_dataset.py first.")
        return

    print(f"Training YOLO classification model on dataset: {dataset_path}")

    # Train the model on the dataset for 3 epochs with image size 224
    results = model.train(data=dataset_path, epochs=3, imgsz=224)

    # Evaluate the model's performance on the validation set
    metrics = model.val()
    print("Validation metrics:", metrics)

    # Test inference on a sample image if available
    sample_image = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/metal_samples.jpg'))
    if os.path.exists(sample_image):
        print("Running inference on sample image...")
        results = model(sample_image)
        for r in results:
            print(f"Inference complete: top class index {r.probs.top1}")
            
    # Export the model to ONNX format
    print("Exporting model...")
    success = model.export(format="onnx")

if __name__ == "__main__":
    main()
