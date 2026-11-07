from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, ToTensord,
    EnsureChannelFirstd, RandFlipd, RandRotate90d, RandGaussianNoised,
)
from torch.utils.data import Dataset
import monai.utils
monai.utils.set_determinism(seed=42)

# will change this based on the model we choose
def cls_transform(training:bool):
    common = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
    ]
    aug = [
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'mask'], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
    ] if training else []
    return Compose(common + aug + [ToTensord(keys=['image', 'mask'])])

def make_cls_dataloader(records : list[dict], training : bool, batch_size=2, num_workers=2):
    ds = Dataset(
        records,
        cls_transform(training)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=num_workers)

# use it like this:
# df = pd.Read_CSV(...)
# dataloader = make_class_loader(df[df.split == "train"], True)