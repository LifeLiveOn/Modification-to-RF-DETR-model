import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def is_black_edge_image(img, threshold=15, edge_ratio=0.5):
    """Quickly check if left or right 10% edges are mostly black."""
    np_img = np.array(img.convert("L"))  # grayscale
    h, w = np_img.shape
    edge_width = max(1, int(w * 0.1))
    left_ratio = np.mean(np_img[:, :edge_width] < threshold)
    right_ratio = np.mean(np_img[:, -edge_width:] < threshold)
    return left_ratio > edge_ratio or right_ratio > edge_ratio


def slice_single_image(
    image_path,
    anns,
    ann_id_start,
    next_img_id,
    tile_size=640,
    overlap=0.0,
    visualize=False,
    black_threshold=15,
    output_dir="tiles/",
    data_type="train"
):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    step = int(tile_size * (1 - overlap))

    new_images, new_annotations = [], []
    ann_id, img_id = ann_id_start, next_img_id

    img_out_dir = os.path.join(output_dir, data_type)
    vis_out_dir = os.path.join(
        output_dir, "visualized", data_type) if visualize else None
    os.makedirs(img_out_dir, exist_ok=True)
    if vis_out_dir:
        os.makedirs(vis_out_dir, exist_ok=True)

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1, y1 = x0 + tile_size, y0 + tile_size
            if x1 > w or y1 > h:
                # skip tiles exceeding bounds (avoid black padding)
                continue

            tile = image.crop((x0, y0, x1, y1))
            gray = np.array(tile.convert("L"))

            # Skip mostly black tiles quickly
            if np.mean(gray < black_threshold) > 0.25:
                continue

            has_box = False
            tile_filename = f"{Path(image_path).stem}_x{x0}_y{y0}.jpg"

            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                x2, y2 = x + bw, y + bh
                if x2 < x0 or y2 < y0 or x > x1 or y > y1:
                    continue

                new_x, new_y = max(x - x0, 0), max(y - y0, 0)
                new_x2, new_y2 = min(
                    x2 - x0, tile_size), min(y2 - y0, tile_size)
                new_w, new_h = new_x2 - new_x, new_y2 - new_y
                if new_w <= 1 or new_h <= 1:
                    continue

                has_box = True
                new_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [new_x, new_y, new_w, new_h],
                    "area": new_w * new_h,
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": []
                })
                ann_id += 1

            if has_box:
                tile.save(os.path.join(img_out_dir, tile_filename))
                if visualize and vis_out_dir:
                    vis_tile = tile.copy()
                    draw = ImageDraw.Draw(vis_tile)
                    for a in new_annotations[-len(anns):]:
                        draw.rectangle(
                            [a["bbox"][0], a["bbox"][1],
                             a["bbox"][0] + a["bbox"][2],
                             a["bbox"][1] + a["bbox"][3]],
                            outline="red", width=2
                        )
                    vis_tile.save(os.path.join(vis_out_dir, tile_filename))

                new_images.append({
                    "id": img_id,
                    "file_name": tile_filename,
                    "width": tile_size,
                    "height": tile_size
                })
                img_id += 1

    return new_images, new_annotations, ann_id, img_id


def slice_folder_images(
    image_dir,
    anno_path,
    output_dir,
    tile_size=640,
    overlap=0.0,
    visualize=False,
    data_type="train"
):
    os.makedirs(output_dir, exist_ok=True)

    with open(anno_path, "r") as f:
        coco = json.load(f)

    all_new_images, all_new_annotations = [], []
    next_ann_id, next_img_id = 1, 1
    skipped = 0

    for idx, img_info in enumerate(coco["images"], 1):
        image_path = os.path.join(image_dir, Path(img_info["file_name"]).name)
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        img = Image.open(image_path)
        if is_black_edge_image(img):
            skipped += 1
            print(f"Skipping {Path(image_path).name} (black edges)")
            continue

        anns = [a for a in coco["annotations"]
                if a["image_id"] == img_info["id"]]
        new_imgs, new_anns, next_ann_id, next_img_id = slice_single_image(
            image_path, anns, next_ann_id, next_img_id,
            tile_size=tile_size, overlap=overlap,
            visualize=visualize, output_dir=output_dir, data_type=data_type
        )

        all_new_images.extend(new_imgs)
        all_new_annotations.extend(new_anns)
        print(
            f"[{idx}/{len(coco['images'])}] {Path(image_path).name} → {len(new_imgs)} tiles", flush=True)

    new_coco = {
        "images": all_new_images,
        "annotations": all_new_annotations,
        "categories": coco["categories"]
    }
    out_json = os.path.join(output_dir, f"{data_type}/_annotations.coco.json")
    with open(out_json, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"\n{data_type.upper()} DONE: {len(all_new_images)} tiles, {len(all_new_annotations)} boxes, {skipped} skipped due to black edges.")


if __name__ == "__main__":
    for dt in ["train", "valid", "test"]:
        slice_folder_images(
            image_dir=f"datasets/hail_1/{dt}/",
            anno_path=f"datasets/hail_1/{dt}/_annotations.coco.json",
            output_dir="datasets/hail_1_cropped/",
            tile_size=640,
            overlap=0.2,
            visualize=True,
            data_type=dt
        )
