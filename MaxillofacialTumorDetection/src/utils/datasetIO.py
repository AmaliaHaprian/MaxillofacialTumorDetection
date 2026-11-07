# .dcm files are folders of individual 2D slices per patient
# they should be stacked and converted into a single 3D file (.nii.gs or .nrrd)
from pathlib import Path


# MONAI data loader converts the .nii.gz / .nrrd files into a 3D tensor

# we will standardize everything to .nii.gz (compressed NIfTL)
CLASS_MAP = {
    "AME":0,
    "DC":1,
    "KCOT":2,
    "RC":3
}
def load_dolchid_dataset(dolchid_dir : str):
    """
    The DOLCHID dataset has this folder structure:
    'DOLCHID/'
        'DOLCHID/cbct_image/'
            'DOLCHID/cbct_image/AME_10_CBCT_Image.nii.gz'
            'DOLCHID/cbct_image/AME_11_CBCT_Image.nii.gz'
        'DOLCHID/cbct_label/'
            'DOLCHID/cbct_label/RC_15_CBCT_Label.nii.gz'

    The function scans the DOLCHID flat folder and returns a list of dictionaries containing:
        - the image
        - the mask (label image)
        - the prefix corresponding to the classification label
        - the class id of the classification label

    dolchid_dir : the string path to the root dolchid folder
    """
    supplied_dir = Path(dolchid_dir)

    if not supplied_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {supplied_dir}")

    if (supplied_dir / "cbct_image").is_dir():
        image_dir = supplied_dir / "cbct_image"
        label_dir = supplied_dir / "cbct_mask"
    
    data_dicts = []

    image_files = sorted(list(image_dir.glob('*_CBCT_Image.nii.gz')))
    for img_path in image_files:
        label_path = Path(str(img_path).replace("cbct_image", "cbct_mask").replace("_CBCT_Image.nii.gz", "_CBCT_Label.nii.gz"))
        if not label_path.exists():
            print(f"Warning: Label missing for {img_path.name}")
            continue

        file_prefix = img_path.name.split("_")[0].upper()
        if file_prefix in CLASS_MAP:
            class_id = CLASS_MAP[file_prefix]

            data_dicts.append({
                "image": str(img_path),
                "mask": str(label_path),
                "class_name": file_prefix,
                "class_id": class_id
            })
        else:
            print(f"Warning: Unknown class prefix '{file_prefix}' in {img_path.name}")

    print(f"Successfully loaded {len(data_dicts)} DOLCHID samples.")
    return data_dicts