import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import seaborn as sns
import numpy as np
import random
from pathlib import Path
import nibabel as nib
from tqdm.auto import tqdm

def read_metadata(records, cache_path=None):
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    rows = []
    for r in tqdm(records, desc="Reading headers"):
        img = nib.load(full_path(r["image"]))
        shape = img.shape
        zooms = img.header.get_zooms()[:3]
        rows.append({
            "case": Path(r["image"]).name,
            "class_name": r.get("class_name"),
            "Shape_X": shape[0], "Shape_Y": shape[1], "Shape_Z": shape[2],
            "Spacing_X": zooms[0], "Spacing_Y": zooms[1], "Spacing_Z": zooms[2],
            "Orientation": "".join(nib.aff2axcodes(img.affine)),
        })
    meta_df = pd.DataFrame(rows)
    if cache_path:
        meta_df.to_csv(cache_path, index=False)
    return meta_df

def sample_intensities(records, voxels_per_case=5000, cache_path=None, seed=42):
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    rng = np.random.default_rng(seed)
    rows = []
    for r in tqdm(records, desc="Sampling intensities"):
        img = np.asarray(nib.load(full_path(r["image"])).dataobj, dtype=np.float32)
        msk = np.asarray(nib.load(full_path(r["mask"])).dataobj) > 0

        for region, vox in (("lesion", img[msk]), ("background", img[~msk])):
            if vox.size == 0:
                continue
            k = min(voxels_per_case, vox.size)
            vals = rng.choice(vox, k, replace=False)
            rows.append(pd.DataFrame({"value": vals, "region": region,
                                      "class_name": r["class_name"]}))
    out = pd.concat(rows, ignore_index=True)
    if cache_path:
        out.to_csv(cache_path, index=False)
    return out

ROOT = Path("/content/drive/MyDrive/cbct-ai/processed/seg")

def full_path(rel):
    return str(ROOT/str(rel).lstrip("/\\"))

def class_distribution(records):
    """
    Creates a bar chart of the class distribution for each split: train, val, test
    """
    df = pd.DataFrame(records)
    grouped_df = df.groupby(['split', 'class_name']).size().unstack(fill_value=0)

    ax = grouped_df.plot(kind="bar", figsize=(10,6), rot=0)

    ax.set_title('Class Distribution per Split Set')
    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Number of Samples')
    plt.legend(title='Class Name')
    plt.tight_layout()
    plt.show()

def volume_stats(records):
    """
    Creates a boxplot of the images volume and offers information about the voxels spacing and image orientation
    """
    meta_df = read_metadata(records,  cache_path="volume_meta.csv")
    fig, axes = plt.subplots(1, 2 , figsize=(10,6))
    sns.boxplot(data = meta_df[['Shape_X', 'Shape_Y', 'Shape_Z']], ax = axes[0])
    axes[0].set_title("CBCT Volume Dimensions")
    axes[0].set_ylabel("Number of Voxels")

    sns.boxplot(data=meta_df[["Spacing_X", "Spacing_Y", "Spacing_Z"]], ax=axes[1])
    axes[1].set_title("Voxel spacing")
    axes[1].set_ylabel("mm")

    plt.tight_layout()
    plt.show()

    print("\n--- Volume Summary Statistics ---")
    print(meta_df[['Spacing_X', 'Spacing_Y', 'Spacing_Z', 'Shape_X', 'Shape_Y', 'Shape_Z']].describe().round(3))

    print("\n--- Anatomical Orientation Consistency ---")
    orientation_counts = meta_df['Orientation'].value_counts()
    for matrix, count in orientation_counts.items():
        print(f"Count: {count} | Direction Matrix: {matrix[:30]}...")

def intensity_stats(records, n=None, voxels_per_patient=5000):
    """
    Creates a histogram of the intensity distribution of the voxels
    """
    hist_df = sample_intensities(records[:10], voxels_per_patient, cache_path="intensity_hist.csv")
    fig, axes = plt.subplots(1, 2 , figsize=(10,6))
    sns.histplot(
        data=hist_df[hist_df.region == "lesion"],
        x = 'value',
        hue = 'class_name',
        bins = 100,
        element = 'step',
        stat = 'density',
        common_norm=False,
        alpha = 0.3,
        ax = axes[0]
    )
    axes[0].set_title("Lesion voxel intensities by class")

    sns.histplot(data=hist_df, x="value", hue="region", bins=100, element="step",
                 stat="density", common_norm=False, alpha=0.3, ax=axes[1])
    axes[1].set_title("Lesion vs. rest of volume")

    for ax in axes:
        ax.set_xlabel("Intensity (CBCT units)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    print(hist_df.groupby("region")["value"].describe(percentiles=[.01, .05, .5, .95, .99]).round(1))


def lesion_stats(records):
    rows=[]
    for r in tqdm(records, desc="Lesion stats"):
        m = np.asarray(nib.load(full_path(r["mask"])).dataobj)>0
        idx =np.where(m)
        if idx[0].size==0:
            rows.append({"case": Path(r["mask"]).name,
                         "class_name": r["class_name"], "volume_ml": 0})
            continue
        ext = [i.max()-i.min()+1 for i in idx]
        rows.append({
            "case": Path(r["mask"]).name,
            "class_name": r["class_name"],
            "volume_ml": m.sum() / 1000,  # 1 voxel = 1 mm^3 at 1 mm spacing
            "ext_x": ext[0], "ext_y": ext[1], "ext_z": ext[2],
            "fraction_of_volume": m.mean()
        })

    out = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=out, x="class_name", y="volume_ml", ax=ax)

    ax.set_title("CBCT Lesion Volume Dimensions Per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Volume")

    plt.tight_layout()
    plt.show()

def show_sample_images(records, num_samples=3, n_slices=5):
    """
    Shows a slice of the 3D image and the overlay between the image and the mask
    """
    samples = random.sample(records, min(num_samples, len(records)))

    fig, axes = plt.subplots(1, len(samples), figsize = (5 * len(samples), 5))

    if len(samples) == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(full_path(sample['image'])))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(full_path(sample['mask'])))
        class_name = sample['class_name']

        z_tumor_areas = np.sum(mask_array > 0, axis=(1,2))

        best_z = np.argmax(z_tumor_areas)
        img_slice = img_array[best_z, :, :]
        mask_slice = mask_array[best_z, :, :]

        axes[i].imshow(img_slice, cmap='gray', vmin=-500, vmax=1500)

        overlay = np.ma.masked_where(mask_slice==0, mask_slice)
        axes[i].imshow(overlay, cmap='autumn', alpha=0.4)

        axes[i].set_title(f"Class: {class_name} | Slice: Z={best_z}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_3d_orthographic_views(sample_dict):
    """
    Finds the 3D center of the tumor and plots the Axial, Coronal,
    and Sagittal planes intersecting that exact point.
    """
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(full_path(sample_dict['image'])))
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(full_path(sample_dict['mask'])))
    class_name = sample_dict['class_name']

    z_coords, y_coords, x_coords = np.where(mask_arr > 0)

    if len(z_coords) == 0:
        print("Warning: No mask found in this volume.")
        return

    center_z = int(np.median(z_coords))
    center_y = int(np.median(y_coords))
    center_x = int(np.median(x_coords))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Orthographic Tumor Views | Class: {class_name} | Center: (X:{center_x}, Y:{center_y}, Z:{center_z})",
                 fontsize=14)

    views = [
        {"title": "Axial (XY Plane)", "img": img_arr[center_z, :, :], "mask": mask_arr[center_z, :, :]},
        {"title": "Coronal (XZ Plane)", "img": img_arr[:, center_y, :], "mask": mask_arr[:, center_y, :]},
        {"title": "Sagittal (YZ Plane)", "img": img_arr[:, :, center_x], "mask": mask_arr[:, :, center_x]}
    ]

    for ax, view in zip(axes, views):
        ax.imshow(view["img"], cmap='gray', vmin=-200, vmax=1500, interpolation="nearest", origin="lower")

        overlay = np.ma.masked_where(view["mask"] == 0, view["mask"])
        ax.imshow(overlay, cmap='autumn', alpha=0.4, interpolation="nearest", origin="lower")

        ax.set_title(view["title"])
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def check_lesion_sizes(records: list[dict]):
    extents = []
    for r in tqdm(records):
        m = np.asarray(nib.load(full_path(r["mask"])).dataobj) > 0
        idx = np.where(m)
        extents.append([i.max() - i.min() + 1 for i in idx])
    extents = np.array(extents)
    print(extents.max(axis=0), np.percentile(extents, 95, axis=0))