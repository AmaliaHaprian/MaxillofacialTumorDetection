import SimpleITK as sitk
from pathlib import Path
import os
import argparse

def convert_dicom_to_nifti(dicom_dir: str, output_dir: str):
    """
    Converts a DICOM directory into a nifti file.
    dicom_dir: the string path to the DICOM directory.
    output_dir: the string path to the output directory where the .nii.gz file will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image_3d = reader.Execute()

    sitk.WriteImage(image_3d, output_path / "image.nii.gz")
    print(f"Converting dicom file {dicom_dir} to nifti in folder {output_dir}")

def convert_nrrd_to_nifti(nrrd_mask_path: str, dicom_reference : str,  output_dir: str):
    """
    Converts a nrrd file to a nifti file.
    nrrd_mask_path: the string path to the nrrd file.
    output_dir: the string path to the output directory where the .nii.gz file will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mask_3d = sitk.ReadImage(nrrd_mask_path)
    image_3d = sitk.ReadImage(dicom_reference)

    if mask_3d.GetDimension() != image_3d.GetDimension():
        raise ValueError(f"Mask and image don't have the same dimensions!")
    aligned_mask = sitk.Resample(
        mask_3d, image_3d, sitk.Transform(image_3d.GetDimension(), sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, mask_3d.GetPixelID()
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(aligned_mask, output_path / "mask.nii.gz")

    print(f"Converting nrrd file {nrrd_mask_path} to nifti in folder {output_dir}")