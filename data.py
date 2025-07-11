from torch.utils.data import Dataset
import os
import torch
from typing import Union
from typing import Tuple
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
from tqdm.autonotebook import tqdm
from PIL import Image
import numpy as np
import torchvision as tv
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.DF = data
        self.mode = mode    # val or train
        # Compose is the callable class which does chain of transformations on the data
        self.to_tensor = tv.transforms.Compose([tv.transforms.ToTensor()])

        # # Consider creating two different transforms based on whether you are in the training or validation dataset.
        # self.val_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
        #                                          tv.transforms.Normalize(mean=train_mean, std=train_std)])
        # self.train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(),tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std), ])

        self.images= self.DF.iloc[:, 0]
        self.labels = self.DF.iloc[:, 1:]
        self.count = 0


    def __len__(self):
        #if self.mode == 'train':
        return len(self.DF)
        #if self.mode == 'val':
        #    return len(self.images.index)

    """The file names and the corresponding labels are listed in the csv-file data.csv. 
    Each row in the csv-file contains the path to an image in our dataset and two numbers indicating 
    if the solar cell shows a 'crack' and if the solar cell can be considered 'inactive'"""

    def __getitem__(self, index): # used for accessing list items

        if self.mode == 'val':
            #self.count += 1
            #print(self.count)
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)
            # img = self.val_transform(img)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1,2))
            #img = self.to_tensor(img)
            return (img, labels)
        if self.mode == 'train':
            #self.count += 1
            #print(self.count)
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)

            # img = self.train_transform(img)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1, 2))
            #img = self.to_tensor(img)
            return (img, labels)



class DataAugmenter:
    def __init__(self, output_dir="augmented", num_augs=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_augs = num_augs

        self.aug_tfs = [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.2),
        ]
        self.transform = T.Compose(self.aug_tfs)

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        new_rows = []

        for idx, row in df.iterrows():
            filename = Path(row["filename"])
            img = imread(filename)

            # Convert to RGB if needed
            if img.ndim == 2 or img.shape[-1] == 1:
                img = gray2rgb(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

            for i in range(self.num_augs):
                img_aug = self.transform(img)

                # Create new filename
                stem = filename.stem
                suffix = filename.suffix if filename.suffix else ".png"
                new_name = f"{stem}_aug{i+1}{suffix}"
                save_path = self.output_dir / new_name

                # Save image
                img_aug.save(save_path)

                # Append new row
                new_rows.append({
                    "filename": str(save_path),
                    "crack": row["crack"],
                    "inactive": row["inactive"],
                    "class_combo": row["class_combo"]
                })

        return pd.DataFrame(new_rows)



def create_dataset(
    data_csv="",
    total_samples=500,
    num_augs={"0_0": -1, "0_1": 49, "1_0": 2, "1_1": 5},
    output_dir="augmented",
    output_train="train.csv",
    output_val="val.csv"
):
    from data import DataAugmenter
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # load and split data
    df = pd.read_csv(data_csv, sep=";")
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    # add class combo column to df
    train["class_combo"] = train.apply(lambda row: f"{row['crack']}_{row['inactive']}", axis=1)
    class_combos = ["0_0", "0_1", "1_0", "1_1"]

    train_balanced = pd.DataFrame()

    for class_combo in class_combos:
        train_class = train[train["class_combo"] == class_combo].copy()

        if num_augs.get(class_combo, -1) > 0:
            augmenter = DataAugmenter(output_dir=output_dir, num_augs=num_augs[class_combo])
            new_rows = augmenter.augment(train_class)
            train_class = pd.concat([train_class, new_rows], ignore_index=True)

        # sample
        train_class = train_class.sample(n=total_samples, replace=True, random_state=42)
        train_balanced = pd.concat([train_balanced, train_class], ignore_index=True)

    # save csv
    train_balanced.to_csv(output_train, index=False, sep=";")
    val.to_csv(output_val, index=False, sep=";")

    print(f"Train set saved to {output_train} ({len(train_balanced)} samples)")
    print(f"Val set saved to {output_val} ({len(val)} samples)")

    return train_balanced, val


