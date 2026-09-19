"""
capture_images.py

Simple webcam / Raspberry Pi camera capture tool for building a
waste-classification dataset (organic, metal, plastic).

USAGE:
    python3 capture_images.py --class organic
    python3 capture_images.py --class metal
    python3 capture_images.py --class plastic

CONTROLS (while the camera window is open):
    s   -> save the current frame to disk
    q   -> quit

Images are saved into:  dataset/raw/<class_name>/<class_name>_XXXX.jpg

Works with:
- A regular USB webcam or laptop camera (OpenCV's default backend)
- A Raspberry Pi camera module IF you already have it exposed as
  /dev/video0 (e.g. via `libcamera` + `v4l2loopback`, or a USB webcam
  plugged into the Pi). This script does not use picamera2 directly
  to keep it portable across laptop/Pi -- see the note at the bottom
  if you want a picamera2-native version instead.
"""

import argparse
import os
import time

import cv2

VALID_CLASSES = ["organic", "metal", "plastic"]


def get_next_index(class_dir: str, class_name: str) -> int:
    """Find the next free image index so re-running the script doesn't
    overwrite previously captured images."""
    existing = [
        f for f in os.listdir(class_dir)
        if f.startswith(class_name) and f.endswith(".jpg")
    ]
    if not existing:
        return 0
    numbers = []
    for f in existing:
        try:
            num = int(f.replace(f"{class_name}_", "").replace(".jpg", ""))
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 0


def main():
    parser = argparse.ArgumentParser(description="Capture labeled images for the waste dataset.")
    parser.add_argument(
        "--class", dest="class_name", required=True, choices=VALID_CLASSES,
        help="Which class you are currently photographing.",
    )
    parser.add_argument(
        "--camera", dest="camera_index", type=int, default=0,
        help="Camera device index (default 0). Try 1 or 2 if 0 doesn't work.",
    )
    parser.add_argument(
        "--out", dest="out_root", default="dataset/raw",
        help="Root output folder (default: dataset/raw)",
    )
    args = parser.parse_args()

    class_dir = os.path.join(args.out_root, args.class_name)
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera_index}. "
              f"Try a different --camera value.")
        return

    next_idx = get_next_index(class_dir, args.class_name)
    saved_count = 0

    print(f"\nCapturing for class: '{args.class_name}'")
    print(f"Saving to: {class_dir}")
    print("Press 's' to save a frame, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to grab frame from camera.")
            break

        display = frame.copy()
        cv2.putText(
            display, f"class: {args.class_name}  saved: {saved_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("Dataset Capture - press 's' to save, 'q' to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            filename = f"{args.class_name}_{next_idx:04d}.jpg"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")
            next_idx += 1
            saved_count += 1
            time.sleep(0.15)  # small debounce so one keypress = one image
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Captured {saved_count} new images for class '{args.class_name}'.")


if __name__ == "__main__":
    main()


