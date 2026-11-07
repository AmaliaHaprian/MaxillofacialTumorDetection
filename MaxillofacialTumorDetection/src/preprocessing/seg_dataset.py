import monai.utils
from monai.data import DataLoader
from monai.transforms import (
    Compose, ScaleIntensityRanged, ToTensord,
    LoadImaged, EnsureChannelFirstd, RandCropByPosNegLabeld, SpatialPadd
)
from tensorflow.python.ops.gen_dataset_ops import CacheDataset

monai.utils.set_determinism(seed=42)
SEG_SIZE = (96, 96, 96)
# MONAI provides two parallel sets of transforms: Array-based (standard) and Dictionary-based (ending in "d" or "D").
def seg_transform(training: bool):
    common = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
    ]
    if not training:
        return Compose(common + [ToTensord(keys = ["image", "mask"])])

    return Compose(common + [
        SpatialPadd(keys=["image", "mask"], spatial_size=SEG_SIZE),
        RandCropByPosNegLabeld(keys=["image", "mask"], label_key="mask", spatial_size=SEG_SIZE, pos=1, neg=1, num_samples=1),
        # Si astea pt augmentare?
        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        # RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
        ToTensord(keys=["image", "mask"]),
    ])

def make_seg_loader(records: list[dict], training: bool, batch_size: 1, num_workers: 2):
    ds = CacheDataset(
        data=records,
        transform=seg_transform(training),
        cache_rate = 0.0 if training else 1.0,
        num_workers=num_workers,
    )
    return DataLoader(ds, batch_size=batch_size if training else 1, shuffle=training, num_workers=num_workers)

# use it like this:
# splits = pd.Read_CSV(...)
# dataloader = make_seg_loader(df[df.split == "train"], True)