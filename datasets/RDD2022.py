# mostly copy-paste from coco.py

from pathlib import Path

from datasets.coco import CocoDetection, make_coco_transforms


def build(image_set, args):
    images_path = Path("/data/datasets/tdt17/RDD2022/ALL/images")
    annotations_path = Path("/data/other/mathiawo/RDD2022_COCO/annotations/")

    assert images_path.exists(), f"images path {images_path} does not exist"
    assert (
        annotations_path.exists()
    ), f"annotations path {annotations_path} does not exist"

    PATHS = {
        "train": (images_path, annotations_path / "train.json"),
        "val": (images_path, annotations_path / "val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
    )
    return dataset
