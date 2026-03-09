import json
from pathlib import Path


def convert_one_coco_json(json_path, output_label_dir, category_mapping=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not images:
        raise ValueError(f"No images found in {json_path}")
    if not categories:
        raise ValueError(f"No categories found in {json_path}")

    # Build category mapping once, reuse across splits
    if category_mapping is None:
        sorted_cats = sorted(categories, key=lambda x: x["id"])
        category_mapping = {cat["id"]: i for i, cat in enumerate(sorted_cats)}
    else:
        sorted_cats = sorted(categories, key=lambda x: x["id"])

    image_map = {img["id"]: img for img in images}

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    output_label_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        img_w = img["width"]
        img_h = img["height"]

        txt_name = Path(file_name).stem + ".txt"
        txt_path = output_label_dir / txt_name

        lines = []
        for ann in ann_by_image.get(image_id, []):
            if "bbox" not in ann:
                continue

            cat_id = ann["category_id"]
            if cat_id not in category_mapping:
                continue

            # COCO format: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # clamp just in case
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            cls_id = category_mapping[cat_id]
            lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return category_mapping, sorted_cats


def convert_dataset(root_dir):
    root = Path(root_dir)
    splits = ["train", "valid", "test"]

    labels_root = root / "labels"
    category_mapping = None
    categories_for_names = None

    for split in splits:
        json_path = root / split / "_annotations.coco.json"
        if not json_path.exists():
            print(f"Skipping {split}: {json_path} not found")
            continue

        output_label_dir = labels_root / split
        category_mapping, sorted_cats = convert_one_coco_json(
            json_path,
            output_label_dir,
            category_mapping=category_mapping
        )

        if categories_for_names is None:
            categories_for_names = sorted_cats

        print(f"Done: {split} -> {output_label_dir}")

    # Save class names once
    if categories_for_names:
        classes_path = labels_root / "classes.txt"
        labels_root.mkdir(parents=True, exist_ok=True)
        with open(classes_path, "w", encoding="utf-8") as f:
            for cat in categories_for_names:
                f.write(cat["name"] + "\n")
        print(f"Saved classes: {classes_path}")


if __name__ == "__main__":
    convert_dataset("merged_annotations")
