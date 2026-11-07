from collections import Counter
from pathlib import Path
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data_dicts, output_csv_path):
    df = pd.DataFrame(data=data_dicts)
    counts = Counter(df["class_id"].astype(int))

    try:
        train_val, test = train_test_split(df, test_size=0.15, stratify=df["class_id"], random_state=42)

        train, val = train_test_split(train_val, test_size=0.15, stratify=train_val["class_id"], random_state=42)
    except ValueError as e:
        raise ValueError(f"Unable to create stratified splits. Class counts: {dict(counts)}") from e


    df["split"] ="train"
    df.loc[val.index, 'split'] = 'val'
    df.loc[test.index, 'split'] = 'test'

    df["case_id"]=df["image"].apply(lambda x: Path(x).name.replace("_CBCT_Image.nii.gz", ""))

    metadata_df = df.copy()
    metadata_df['image'] = metadata_df['image'].apply(lambda x: f"cbct_image/{Path(x).name}")
    metadata_df['mask'] = metadata_df['mask'].apply(lambda x: f"cbct_mask/{Path(x).name}")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cols = ["case_id", "image", "mask", "class_name", "class_id", "split"]
    metadata_df[cols].to_csv(output_csv_path, index=False)

    print(f"Total Patients: {len(metadata_df)}")
    print(f"Training set:   {len(train)} patients")
    print(f"Validation set: {len(val)} patients")
    print(f"Test set: {len(test)} patients")

    splits = {
        'train': df[df['split'] == 'train'].to_dict(orient='records'),
        'val': df[df['split'] == 'val'].to_dict(orient='records'),
        'test': df[df['split'] == 'test'].to_dict(orient='records')
    }
    return splits
