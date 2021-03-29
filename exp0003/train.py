####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, KFold
import segmentation_models_pytorch as smp
from catalyst.contrib.nn.criterion.dice import DiceLoss

#from sklearn import model_selection
import albumentations as A
import timm
import glob
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from PIL import Image

cv2.setNumThreads(0)
####################
# Utils
####################
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

####################
# Config
####################

conf_dict = {'batch_size': 8,#32, 
             'epoch': 30,
             'image_size': 128,#640,
             'image_scale': 2,
             'encoder_name': 'timm-efficientnet-b0',
             'lr': 0.001,
             'fold': 0,
             'csv_path': '../input/extract-test/train.csv',
             'data_dir': '../input/extract-test/size_2048',
             'output_dir': './',
             'use_mask_exist': True,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################

class HuBMAPDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, "image"]
        mask_path = self.data.loc[idx, "mask"]

        # [TODO] 画像読み込みをpytorch nativeにしたい
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        trans = self.transform(image=image, mask=mask)
        image = torch.from_numpy(trans["image"].transpose(2, 0, 1))
        mask = torch.from_numpy(trans["mask"]).unsqueeze(dim=0).float()
        
        return image, mask
           
####################
# Data Module
####################

class HuBMAPDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf      

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(self.conf.csv_path)
            
            kf = KFold(n_splits=5, shuffle=True, random_state=2021)
            for fold, (train_index, val_index) in enumerate(kf.split(df.values)):
                df.loc[val_index, "fold"] = int(fold)
            df["fold"] = df["fold"].astype(int)
                
            train_df = df[df['fold'] != self.conf.fold]
            valid_df = df[df['fold'] == self.conf.fold]
            
            train_image_list = []
            for index, row in train_df.iterrows():
                train_image_list += glob.glob(os.path.join(self.conf.data_dir, "*" + row['id'] + "_image.png"))
            train_df = pd.DataFrame({'image': train_image_list})
            train_df['mask'] = train_df['image'].str[:-9]+'mask.png'
            
            if self.conf.use_mask_exist:
                print('check no mask image')
                remove_index = []
                for index, row in tqdm(train_df.iterrows()):
                    if np.sum(cv2.imread(row['mask'])) == 0:
                        remove_index.append(index)
                train_nomask_df = train_df.loc[remove_index]
                train_df = train_df.drop(remove_index)
                print(len(train_nomask_df))#1950
                print(len(train_df))#1108
            
            valid_image_list = []
            for index, row in valid_df.iterrows():
                valid_image_list += glob.glob(os.path.join(self.conf.data_dir, "*" + row['id'] + "_image.png"))
            valid_df = pd.DataFrame({'image': valid_image_list})
            valid_df['mask'] = valid_df['image'].str[:-9]+'mask.png'


            train_transform = A.Compose([
                        A.RandomCrop(height=self.conf.image_size*self.conf.image_scale, width=self.conf.image_size*self.conf.image_scale, p=1), 
                        A.Resize(height=self.conf.image_size, width=self.conf.image_size, p=1),
                        A.Flip(p=0.5),
                        A.ShiftScaleRotate(p=0.5),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                        A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                        A.CLAHE(clip_limit=(1,4), p=0.5),
                        A.OneOf([
                            A.OpticalDistortion(distort_limit=1.0),
                            A.GridDistortion(num_steps=5, distort_limit=1.),
                            A.ElasticTransform(alpha=3),
                        ], p=0.50),
                        A.OneOf([
                            A.GaussNoise(var_limit=[10, 50]),
                            A.GaussianBlur(),
                            A.MotionBlur(),
                            A.MedianBlur(),
                        ], p=0.50),
                        #A.Resize(size, size),
                        A.OneOf([
                            A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                            A.Downscale(scale_min=0.75, scale_max=0.95),
                        ], p=0.5),
                        A.IAAPiecewiseAffine(p=0.5),
                        A.IAASharpen(p=0.5),
                        A.Cutout(max_h_size=int(self.conf.image_size * 0.1), max_w_size=int(self.conf.image_size * 0.1), num_holes=5, p=0.5),
                        A.Normalize()
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=int(2048/self.conf.image_scale), width=int(2048/self.conf.image_scale), interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            self.train_dataset = HuBMAPDataset(train_df, transform=train_transform)
            self.train_nomask_dataset = HuBMAPDataset(train_nomask_df, transform=train_transform)
            self.valid_dataset = HuBMAPDataset(valid_df, transform=valid_transform)
            
        elif stage == 'test':
            pass
        
    def train_dataloader(self):
        #mask_batch = int(self.conf.batch_size*0.8)
        #no_mask_batch = self.conf.batch_size - int(self.conf.batch_size*0.8)
        return [DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True),
                DataLoader(self.train_nomask_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)]

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)


####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = smp.Unet(encoder_name=conf.encoder_name, in_channels=3, classes=1)
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.diceloss = DiceLoss()
        #self.diceloss = smp.utils.losses.DiceLoss(activation='sigmoid')
        self.dice =  smp.utils.losses.DiceLoss(activation='sigmoid')

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch[0]
        x2, y2 = batch[1]
        #x, y = torch.cat([x1, x2], dim=0), torch.cat([y1, y2], dim=0) 
        
        # cutmix
        lam = np.random.beta(0.5, 0.5)
        rand_index = torch.randperm(x.size()[0]).type_as(x).long()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        y[:, :, bbx1:bbx2, bby1:bby2] = y[rand_index, :, bbx1:bbx2, bby1:bby2]

        # mixnoise
        lam = np.minimum(np.random.beta(0.5, 0.5), 0.25)
        x = lam * x + (1 - lam) * x2
        
        
        y_hat = self.model(x)
        #loss = self.diceloss(y_hat, y) + self.bceloss(y_hat, y)
        loss = self.bceloss(y_hat, y)
        
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        #loss = self.diceloss(y_hat, y) + self.bceloss(y_hat, y)
        loss = self.bceloss(y_hat, y)
        dice = 1-self.dice(y_hat, y)
        
        return {
            "val_loss": loss,
            "val_dice": dice
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_dice = torch.stack([x["val_dice"] for x in outputs]).mean()

        self.log('val_loss', avg_val_loss)
        self.log('val_dice', avg_val_dice)
        
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_dice', 
                                          save_last=True, save_top_k=5, mode='max', 
                                          save_weights_only=True, filename='{epoch}-{val_dice:.5f}')

    data_module = HuBMAPDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        multiple_trainloader_mode='min_size',
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()
