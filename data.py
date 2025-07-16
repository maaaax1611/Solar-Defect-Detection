from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
from tqdm.autonotebook import tqdm
from PIL import Image
import numpy as np
import torchvision as tv
import torchvision.transforms as T
import pandas as pd
from typing import Union
from typing import Tuple
from tqdm.autonotebook import tqdm
from PIL import Image


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    """
    Custom PyTorch dataset for solar panel defect detection.

    Args
    ----
    data : pandas.DataFrame
        Should contain at least the columns "image_path" and "label".
    mode : str
        Either "train" or "val", determines whether augmentations are applied.
    """

    def __init__(self, data: pd.DataFrame, mode: str = "train") -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.data = data.reset_index(drop=True)
        self.mode = mode

        # 1) Define image transformations
        pil_tfm = [
            T.ToPILImage()
        ]

        base_tfms = [
            T.ToTensor(),
            T.Normalize(mean=train_mean, std=train_std),
        ]

        if mode == "train":
            # Apply data augmentation in training mode
            # maybe add opening + closing and gaussian blurring here
            aug_tfms = [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.1)
            ]
            self.transform = T.Compose(pil_tfm + aug_tfms + base_tfms)
        else:
            # No augmentation in validation mode
            self.transform = T.Compose(pil_tfm + base_tfms)

    def __len__(self) -> int:
        # Return number of samples
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get row from DataFrame
        record = self.data.iloc[index]

        # Load image from file
        img_path = Path(record["filename"])
        img = imread(img_path)

        # Convert grayscale image to RGB if necessary
        if img.ndim == 2 or img.shape[-1] == 1:
            img = gray2rgb(img)

        # Ensure image is in uint8 format (required by ToPILImage)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

        # Load label and convert to tensor
        label_crack: Union[int, list, np.ndarray] = record["crack"]
        label_inactive: Union[int, list, np.ndarray] = record["inactive"]
        label_tensor = torch.as_tensor((label_crack, label_inactive), dtype=torch.long)

        # Apply transformations to image
        img_tensor = self.transform(img)

        return img_tensor, label_tensor
    

import pandas as pd

class DatasetUpsampler:
    def __init__(self, df, class_columns, target_counts):
        """
        Initializes the upsampler.

        Args:
            df (pd.DataFrame): The input dataset (e.g., training set).
            class_columns (list): List of column names that define the class combination (e.g., ['crack', 'inactive']).
            target_counts (dict): A dictionary mapping class combinations (as strings, e.g., '0_1') to desired sample counts.
        """
        self.df = df.copy()
        self.class_columns = class_columns
        self.target_counts = target_counts

        # Create a new column with combined class labels, e.g., "0_1"
        self.df["class_combo"] = self.df[class_columns].astype(str).agg("_".join, axis=1)

    def upsample(self):
        """
        Performs upsampling (or downsampling) based on the provided target_counts.

        Returns:
            pd.DataFrame: A new DataFrame containing the balanced dataset.
        """
        balanced_dfs = []

        for combo, target_count in self.target_counts.items():
            # Select all rows with the current class combination
            df_subset = self.df[self.df["class_combo"] == combo]

            if df_subset.empty:
                print(f"Class '{combo}' not found in dataset - skipping.")
                continue

            # Sample with or without replacement depending on availability
            if len(df_subset) >= target_count:
                sampled = df_subset.sample(n=target_count, replace=False, random_state=42)
            else:
                sampled = df_subset.sample(n=target_count, replace=True, random_state=42)

            balanced_dfs.append(sampled)

        # Combine all upsampled subsets into a single DataFrame
        return pd.concat(balanced_dfs, ignore_index=True)