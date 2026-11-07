import numpy as np
import pandas as pd
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                              Spacingd, CropForegroundd, ResizeWithPadOrCropd)
from pathlib import Path
import nibabel as nib
import monai.utils
monai.utils.set_determinism(seed=42)

RAW= Path("/content/drive/MyDrive/DOLCHID_140")
PROCESSED = Path("/content/drive/MyDrive/cbct-ai/processed")
SPLITS = [Path("/content/drive/MyDrive/cbct-ai/metadata.csv")]
CLS_SIZE = (96,96,96)
MARGIN = 8

base = [
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Orientationd(keys=["image", "mask"], axcodes="RAS"),
    Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
]

def save_base_transforms(res, img_out, mask_out):
    affine = res["image"].affine.numpy() if hasattr(res["image"], "affine") else np.eye(4)
    img = res["image"][0].numpy().astype(np.float32)
    msk = (res["mask"][0].numpy() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(img, affine), img_out)
    nib.save(nib.Nifti1Image(msk, affine), mask_out)

seg_pipeline = Compose(base)
cls_pipeline = Compose(base + [
    CropForegroundd(keys=["image", "mask"], source_key="mask", margin=MARGIN),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=CLS_SIZE, mode="constant", constant_values=-1000),
    ResizeWithPadOrCropd(keys=["mask"], spatial_size=CLS_SIZE, mode="constant", constant_values=0),
    ])
pipes = {
    "seg": seg_pipeline,
    "cls": cls_pipeline,
}

def main():
    df = pd.concat([pd.read_csv(p) for p in SPLITS]).drop_duplicates(subset="image")
    for sub in pipes:
        (PROCESSED / sub).mkdir(parents=True, exist_ok=True)
        (PROCESSED / sub).mkdir(parents=True, exist_ok=True)

    for row in df.itertuples():
        src_image = RAW / str(row.image).lstrip("/\\")
        src_mask = RAW / str(row.mask).lstrip("/\\")

        if not (src_image.exists() and src_mask.exists()):
            print(f"SKIP : missing image or mask")
            continue

        for sub, pipe in pipes.items():
            img_out = PROCESSED / sub / row.image
            msk_out = PROCESSED / sub / row.mask

            img_out.parent.mkdir(parents=True, exist_ok=True)
            msk_out.parent.mkdir(parents=True, exist_ok=True)

            if img_out.exists() and msk_out.exists():  # resume after a disconnect
                continue

            res = pipe({"image": str(src_image), "mask": str(src_mask)})
            if res["mask"].sum() == 0:
                print(f"WARNING: empty mask for ({sub})")

            save_base_transforms(res, img_out, msk_out)
            print(f"[{sub}] saved")

if __name__ == "__main__":
    main()