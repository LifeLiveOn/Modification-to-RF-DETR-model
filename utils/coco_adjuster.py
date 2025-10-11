import json
from pathlib import Path


def remap_coco_categories(json_path: str, num_classes: int = 1, keep_names=None, output_suffix=""):
    """
    Adjust COCO annotation categories for a given num_classes setup.

    Args:
        json_path (str): Path to input _annotations.coco.json
        num_classes (int): Desired number of classes in the model (1 for hail-only, etc.)
        keep_names (list[str] or None): Category names to keep (optional)
        output_suffix (str): Suffix for the output JSON file
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"⚠️ Skipping: {json_path} not found")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"\n📂 Processing: {json_path}")
    print(
        f"   Images: {len(data['images'])}, Annotations: {len(data['annotations'])}")

    # --- Filter and remap categories ---
    categories = data["categories"]
    cat_name_map = {c["id"]: c["name"] for c in categories}

    if keep_names:
        keep_ids = [c["id"] for c in categories if c["name"] in keep_names]
    else:
        keep_ids = [c["id"] for c in categories]

    keep_ids = keep_ids[:num_classes]
    id_remap = {old_id: new_id for new_id,
                old_id in enumerate(sorted(keep_ids))}
    new_categories = [
        {"id": new_id, "name": cat_name_map[old_id], "supercategory": "none"}
        for old_id, new_id in id_remap.items()
    ]

    # --- Filter and remap annotations ---
    new_annotations = []
    dropped = 0
    for ann in data["annotations"]:
        old_cat = ann["category_id"]
        if old_cat not in id_remap:
            dropped += 1
            continue
        ann["category_id"] = id_remap[old_cat]
        new_annotations.append(ann)

    new_data = data.copy()
    new_data["annotations"] = new_annotations
    new_data["categories"] = new_categories

    # --- Write output ---
    out_path = json_path.with_name(json_path.stem + f"{output_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Saved: {out_path}")
    print(f"   → Categories: {[c['name'] for c in new_categories]}")
    print(f"   → Dropped {dropped} annotations\n")


if __name__ == "__main__":
    # === PARAMETERS ===
    dataset_root = Path("merged_annotations")   # path to your dataset root
    num_classes = 1                             # hail-only mode
    keep_names = ["hail"]                       # categories to keep

    for split in ["train", "valid", "test"]:
        json_file = dataset_root / split / "_annotations.coco.json"
        remap_coco_categories(
            json_path=json_file,
            num_classes=num_classes,
            keep_names=keep_names,
            output_suffix=""
        )

    print("🏁 Done remapping all splits.")
