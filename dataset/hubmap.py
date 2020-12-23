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
import numpy as np

cv2.setNumThreads(0)

def get_dataset2(cfg: DictConfig) -> dict:
    
    df = pd.read_csv(to_absolute_path(os.path.join(cfg.data_root, 'train.csv')))#[['id']]

    kf = load_obj(cfg.cv_split.class_name)(**cfg.cv_split.params)
    for fold, (train_index, val_index) in enumerate(kf.split(df.values)):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)

    train_df = df[df["fold"] != cfg.target_fold]
    valid_df = df[df["fold"] == cfg.target_fold]

    valid_img_id = list(valid_df['id'].values)

    train_image_list = []
    for index, row in train_df.iterrows():
        train_image_list += glob.glob(to_absolute_path(os.path.join(cfg.data_dir, "*" + row['id'] + "_image.png")))
    if cfg.ext_data_dir is not None:
        ext_img_list = glob.glob(to_absolute_path(os.path.join(cfg.ext_data_dir, "*_image.png")))
        for i in ['HBM227.QKNQ.293', 'HBM345.LXHZ.233', 'HBM635.BJXT.387', 'HBM676.TDHK.358', 'HBM662.PBGS.268', 'HBM464.GFFC.829', 'HBM385.RWPR.397', 'HBM894.GBWP.856']:
            ext_img_list = [im for im in ext_img_list if i not in im]
        train_image_list += ext_img_list

    train_df = pd.DataFrame({'image': train_image_list})
    train_df['mask'] = train_df['image'].str[:-9]+'mask.png'
    if cfg.use_mask_exist:
        remove_index = []
        for index, row in train_df.iterrows():
            if np.sum(cv2.imread(row['mask'])) == 0:
                remove_index.append(index)
        train_df = train_df.drop(remove_index)

    valid_image_list = []
    for index, row in valid_df.iterrows():
        valid_image_list += glob.glob(to_absolute_path(os.path.join(cfg.data_dir, "*" + row['id'] + "_image.png")))
    valid_df = pd.DataFrame({'image': valid_image_list})
    valid_df['mask'] = valid_df['image'].str[:-9]+'mask.png'

    #valid_df['id'] = to_absolute_path(cfg.data_root) + '/train/' + valid_df['id'] + '.tiff'
    #print(train_df)
    #print(valid_df)

    train_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.train]
    train_augs = A.Compose(train_augs_list)

    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.valid]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = HuBMAPDataset(train_df, transform=train_augs)
    #valid_dataset = HuBMAPValidDataset(valid_df, transform=valid_augs)
    valid_dataset = HuBMAPDataset(valid_df, transform=valid_augs)

    return {"train": train_dataset, "valid": valid_dataset, 'valid_img_id': valid_img_id}
    

class HuBMAPValidDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {'img_pth':self.data.loc[idx, "id"], 'mask':self.data.loc[idx, "encoding"]}


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

    valid_augs_list = [load_obj(i["class_name"])(**i["params"]) for i in cfg.augmentation.valid]
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
        mask = torch.from_numpy(trans["mask"]).unsqueeze(dim=0).float()
        
        return image, mask
