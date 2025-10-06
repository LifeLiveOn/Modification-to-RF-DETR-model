import os
import json
from pathlib import Path


def merge_coco_datasets(
    json1,
    json2,
    output_json,
    dataset1="ds2/train",
    dataset2="tiles_dataset/train"
):
    # --- Load datasets ---
    with open(json1, "r") as f:
        coco1 = json.load(f)
    with open(json2, "r") as f:
        coco2 = json.load(f)

    merged = {
        "images": [],
        "annotations": [],
        "categories": coco1["categories"],  # assume same categories
    }

    # --- Track max IDs ---
    max_img_id = max((img["id"] for img in coco1["images"]), default=0)
    max_ann_id = max((ann["id"] for ann in coco1["annotations"]), default=0)

    # --- Copy dataset1 ---
    for img in coco1["images"]:
        new_img = img.copy()
        new_img["file_name"] = f"{dataset1}/{img['file_name']}".replace(
            "\\", "/")
        merged["images"].append(new_img)
    merged["annotations"].extend(coco1["annotations"])

    # --- Merge dataset2 safely ---
    img_name_to_newid = {}
    for img in coco2["images"]:
        max_img_id += 1
        new_img = img.copy()
        new_img["id"] = max_img_id
        new_img["file_name"] = f"{dataset2}/{new_img['file_name']}".replace(
            "\\", "/")
        img_name_to_newid[img["file_name"]] = new_img["id"]
        merged["images"].append(new_img)

    # --- Relink annotations by filename, not image_id ---
    for ann in coco2["annotations"]:
        old_img = next(
            (img for img in coco2["images"] if img["id"] == ann["image_id"]), None)
        if not old_img:
            continue
        old_name = old_img["file_name"]
        new_img_id = img_name_to_newid.get(old_name)
        if not new_img_id:
            continue

        max_ann_id += 1
        new_ann = ann.copy()
        new_ann["id"] = max_ann_id
        new_ann["image_id"] = new_img_id
        merged["annotations"].append(new_ann)

    # --- Save final merged dataset ---
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged dataset saved to {output_json}")
    print(f"  Total images: {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    print(f"  Categories: {len(merged['categories'])}")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        merge_coco_datasets(
            json1=f"datasets/hail_1_cropped/{split}/_annotations.coco.json",
            json2=f"datasets/hail_2/{split}/_annotations.coco.json",
            output_json=f"merged_annotations/{split}/_annotations.coco.json",
            dataset1=f"datasets/hail_1_cropped/{split}",
            dataset2=f"datasets/hail_2/{split}",
        )
