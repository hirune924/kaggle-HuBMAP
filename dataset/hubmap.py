from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import cv2
from hydra.utils import to_absolute_path
import os
import albumentations as A
import pandas as pd
from utils.utils import load_obj
import torch
import glob
from PIL import Image

def get_dataset(cfg: DictConfig) -> dict:
    
    image_list = glob.glob(to_absolute_path(os.path.join(cfg.data_dir, "*_image.png")))

    df = pd.DataFrame({'image': image_list})
    df['mask'] = df['image'].str[:-9]+'mask.png'

    kf = load_obj(cfg.cv_split.class_name)(**cfg.cv_split.params)
    for fold, (train_index, val_index) in enumerate(kf.split(df.values)):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)

    train_df = df[df["fold"] != cfg.target_fold]
    valid_df = df[df["fold"] == cfg.target_fold]

    train_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.train]
    train_augs = A.Compose(train_augs_list)

    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.train]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = HuBMAPDataset(train_df, transform=train_augs)
    valid_dataset = HuBMAPDataset(valid_df, transform=valid_augs)

    return {"train": train_dataset, "valid": valid_dataset}
    

class HuBMAPDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = to_absolute_path(self.data.loc[idx, "image"])
        mask_path = to_absolute_path(self.data.loc[idx, "mask"])

        # [TODO] 画像読み込みをpytorch nativeにしたい
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        trans = self.transform(image=image, mask=mask)
        image = torch.from_numpy(trans["image"].transpose(2, 0, 1))
        mask = torch.from_numpy(trans["mask"])
        
        return image, mask
