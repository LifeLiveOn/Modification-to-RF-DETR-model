import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from rfdetr import RFDETRBase
import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning)


def run_training():
    model = RFDETRBase(num_classes=2)
    resume_path = "merged_annotations/output/checkpoint.pth"
    if Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
    else:
        resume_path = None

    model.train(
        dataset_dir="merged_annotations",
        epochs=20,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        num_workers=0,
        output_dir="merged_annotations/output",
        tensorboard=True,
        expanded_scales=True,
        resume=resume_path,
        balance_annotated_unannotated=True,
    )


def run_rfdetr_inference(checkpoint_path: str, image_path: str, class_names=None, save_dir="saved_predictions"):
    """
    Run RF-DETR inference on one image and save visualization using supervision.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading model from {ckpt}")
    model = RFDETRBase(num_classes=len(class_names)
                       if class_names else 1, pretrain_weights=str(ckpt))
    # model.optimize_for_inference()

    # Load image
    image = Image.open(image_path)

    # Run prediction
    detections = model.predict(image, threshold=0.5)

    # Default label names
    if class_names is None:
        class_names = ["damage"]

    # Build labels
    labels = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        if class_id < len(class_names):
            label = f"{class_names[class_id]} {confidence:.2f}"
        else:
            label = f"Unknown({class_id}) {confidence:.2f}"
        labels.append(label)

    # Annotate with supervision
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(image.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    # Save output
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_pred.jpg"
    annotated.save(save_path)

    print(f"Saved annotated image to: {save_path}")
    return detections, str(save_path)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required for Windows
    # run_training()

    run_training()
    # test_img_path = "tiles_sample/images/DJI_0007size12391882_jpg.rf.9cb167f97e5a37e7602f96810c25f224_x1152_y576.jpg"
    # run_rfdetr_inference(
    #     checkpoint_path="ds2/output/checkpoint.pth",
    #     image_path=test_img_path,
    #     class_names=["wind_damage", "hail_damage"],
    #     save_dir="ds2/saved_predictions"
    # )
