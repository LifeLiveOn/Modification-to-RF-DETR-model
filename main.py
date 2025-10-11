import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from rfdetr import RFDETRBase
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)


def run_training(
    num_classes: int = 1,
    path_to_dataset: str = "merged_annotations",
    resume_checkpoint: str | None = None,
    output_dir: str = "merged_annotations/output"
):
    model = RFDETRBase(num_classes=num_classes)

    # Resume path logic
    resume_path = (
        resume_checkpoint
        if resume_checkpoint
        else Path(output_dir) / "checkpoint.pth"
    )
    if Path(resume_path).exists():
        print(f"✅ Resuming from {resume_path}")
    else:
        print("🆕 Starting fresh training...")
        resume_path = None

    model.train(
        dataset_dir=path_to_dataset,
        epochs=30,
        batch_size=8,
        grad_accum_steps=4,
        lr=1e-4,
        num_workers=0,
        output_dir=output_dir,
        tensorboard=True,
        resume=resume_path,
        seed=42,
        early_stopping=True,
        early_stopping_patience=10,
        gradient_checkpointing=True,
    )


def run_rfdetr_inference(model, image_path: str, class_names=None, save_dir="saved_predictions"):
    """Run RF-DETR inference on one image and save visualization using supervision."""
    image = Image.open(image_path)

    detections = model.predict(image, threshold=0.35)
    print("Class IDs:", detections.class_id)
    print("Confidences:", detections.confidence)
    print("Boxes:", detections.xyxy if hasattr(detections, "xyxy") else None)

    if class_names is None:
        class_names = ["damage"]

    labels = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        if class_id < len(class_names):
            label = f"{class_names[class_id]} {confidence:.2f}"
        else:
            label = f"Unknown({class_id}) {confidence:.2f}"
        labels.append(label)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(image.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_pred.jpg"
    annotated.save(save_path)
    print(f"Saved annotated image to: {save_path}")

    return detections, str(save_path)


if __name__ == "__main__":
    import multiprocessing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'], default='test', help='Mode: train or test'
    )
    args = parser.parse_args()
    mode = args.mode

    multiprocessing.freeze_support()  # required for Windows

    class_names = ["hail"]
    if mode == "train":
        checkpoint_path = "merged_annotations/output/checkpoint.pth"
        # HAIL ONLY training

        num_classes = len(class_names)
        run_training(
            num_classes=num_classes,
            path_to_dataset="merged_annotations",
            # or "merged_annotations/output/checkpoint.pth" if continuing
            resume_checkpoint=checkpoint_path,
            output_dir="merged_annotations/output"
        )
    else:
        # === Paths ===
        test_folder_path = r"datasets/hail_2/test"
        # === Inference ===
        checkpoint_path = "merged_annotations/output/checkpoint_best_ema.pth"
        model = RFDETRBase(
            num_classes=len(class_names),
            pretrain_weights=checkpoint_path
        )

        for img in Path(test_folder_path).glob("*.*"):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            run_rfdetr_inference(
                model=model,
                image_path=str(img),
                class_names=class_names,
                save_dir="run/saved_predictions/hail_2"
            )
