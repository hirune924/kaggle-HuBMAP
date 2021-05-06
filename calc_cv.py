from sklearn.model_selection import StratifiedKFold, KFold
import os
import sys

import numpy as np
import pandas as pd
import cv2
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Dataset
import albumentations as albu
import segmentation_models_pytorch as smp
import torch
from tqdm.notebook import tqdm
import glob
# from tqdm.auto import tqdm

from PIL import Image
import tifffile as tiff
import warnings
from collections import OrderedDict
import shutil
import json

import albumentations as A
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import Window

####################
# Config
####################

conf_dict = {
    'fold': 0,
    'data_dir': '/kqi/parent/22020667/raw/',
    'model_dir': '/kqi/parent/22020920/',
    'output': '/kqi/output',
    'last_epoch': True,
    'tile_size': 512*4,
    'margin': 64*4,
    'reduce': 2,
    'bs': 8}
conf_base = OmegaConf.create(conf_dict)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

class Reader(object):
    def __init__(self, filename):
        identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        self.handle = rasterio.open(filename, 'r', **{ 'transform' : identity, 'num_threads' : 'all_cpus' })
        if self.handle.count != 3:
            self.layers = []
            if len(self.handle.subdatasets) > 0:
                for i, subdataset in enumerate(self.handle.subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            
    def get_size(self):
        return self.handle.width, self.handle.height

    def read(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        img = np.zeros([h, w, 3], dtype=np.uint8)
        # 書込先
        px = -x1 if x1 < 0 else 0
        py = -y1 if y1 < 0 else 0
        # 取得元
        u1 = max(0, x1)
        v1 = max(0, y1)
        u2 = min(self.handle.width , x1 + w)
        v2 = min(self.handle.height, y1 + h)
        du = u2 - u1
        dv = v2 - v1
        if self.handle.count != 3:
            for i, layer in enumerate(self.layers):
                img[py:py+dv, px:px+du, i] = layer.read(1, window=Window.from_slices((v1, v2),(u1, u2)))
        else:
            img[py:py+dv, px:px+du, :] = np.moveaxis(self.handle.read([1, 2, 3], window=Window.from_slices((v1, v2),(u1, u2))), 0, -1)
        return img
    
    def read_tile(self, w,h, img_info, tile_size, margin):
        tile = self.read(x1=max(0,w*tile_size-margin),
                           y1=max(0,h*tile_size-margin),
                           x2=(w+1)*tile_size+margin,
                           y2=(h+1)*tile_size+margin)

        if h == 0:
            tile = np.pad(tile,[[margin,0],[0,0],[0,0]],constant_values=0)
        if h == img_info['num_split_h']-1:
            tile = np.pad(tile,[[0,img_info['pad_h']+margin],[0,0],[0,0]],constant_values=0)
        if w == 0:
            tile = np.pad(tile,[[0,0],[margin,0],[0,0]],constant_values=0)
        if w == img_info['num_split_w']-1:
            tile = np.pad(tile,[[0,0],[0,img_info['pad_w']+margin],[0,0]],constant_values=0)
        return tile
    
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

#https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
#with bug fix
def rle_encode_less_memory(img):
    #watch out for the bug
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

# https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/198343
def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



def get_img_info(img_id, tile_size, margin, train=False):
    if train:
        img_name = '/kqi/parent/22020667/raw/train/'+img_id+'.tiff'
    else:
        img_name = '/kqi/parent/22020667/raw/test/'+img_id+'.tiff'
    
    reader = Reader(img_name)
    width, height = reader.get_size()

    pad_w = tile_size - width%tile_size
    pad_h = tile_size - height%tile_size
    num_split_w, num_split_h = int(width/tile_size)+1, int(height/tile_size)+1
    reader.close()
    
    return {'width': width, 'height': height, 'num_split_w': num_split_w, 'num_split_h': num_split_h, 'pad_h': pad_h, 'pad_w': pad_w}

def inference(models, img_id, img_info, tile_size, margin, reduce, train=False):
    TH=0.5
    if train:
        img_name = '/kqi/parent/22020667/raw/train/'+img_id+'.tiff'
    else:
        img_name = '/kqi/parent/22020667/raw/test/'+img_id+'.tiff'
    reader = Reader(img_name)
    
    tile_size2 = int((tile_size+margin*2)/reduce) #640
    transform = A.Compose([A.Resize(height=tile_size2, width=tile_size2, 
                                    interpolation=1, always_apply=True),
                          A.Normalize()])
    
    pad_w, pad_h = img_info['pad_w'], img_info['pad_h']
    
    mask = np.zeros((img_info['height']+pad_h, img_info['width']+pad_w), dtype=np.uint8)
    
    with torch.no_grad():
        for h in range(img_info['num_split_h']):
            for w in range(img_info['num_split_w']):
                #image = np.load('../tiles/{}_w{}_h{}.npy'.format(img_id,w,h))
                image = reader.read_tile(w,h, img_info, tile_size, margin)
                #image = (image/255.0).astype("float32")
                tmp_pred = np.zeros((tile_size, tile_size))
                if np.sum(image)!=0:
                    image = torch.from_numpy(transform(image=image)['image'].transpose(2, 0, 1)).unsqueeze(dim=0).cuda()
                    #images = torch.cat([image, image, image, image], dim=0)
                    images = torch.cat([image], dim=0)
                    #print(images.shape)
                    for m in models:
                        pred_tile = torch.sigmoid(m(images))
                        #pred_tile = m(images)
                        #pred_tile[1] = pred_tile[1]
                        #pred_tile[2] = pred_tile[2]
                        #pred_tile[3] = pred_tile[3]
                        pred_tile = torch.mean(pred_tile, dim=0).cpu().detach().numpy().squeeze()
                        # 必要な中心部分だけ切り出して格納する
                        tmp_pred += cv2.resize(pred_tile,
                                               (tile_size+margin*2,tile_size+margin*2)
                                              )[margin:-margin,margin:-margin]/(len(models))
                
                mask[h*tile_size:(h+1)*tile_size, w*tile_size:(w+1)*tile_size] = tmp_pred>TH
    return mask[:img_info['height'], :img_info['width']]


  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    
    # get target df
    df = pd.read_csv(os.path.join(conf.data_dir, 'train.csv'))
            
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    for fold, (train_index, val_index) in enumerate(kf.split(df.values)):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)
    df["dice"] = df["fold"].astype(float)
    target_df = df[df['fold'] == conf.fold]
    #print(target_df)
    
    # get target model
    if conf.last_epoch:
        MODEL = [os.path.join(conf.model_dir, f'fold{conf.fold}/ckpt/last.ckpt')]
    else:
        target_model = glob.glob(os.path.join(conf.model_dir, f'fold{conf.fold}/ckpt/epoch*.ckpt'))
        scores = [float(os.path.splitext(os.path.basename(i))[0].split('=')[-1]) for i in target_model]
        MODEL = [target_model[scores.index(max(scores))]]
    #print(MODEL)
    
    # build model
    models = []
    for model_path in MODEL:
        model = smp.Unet("timm-efficientnet-b3", 
                     in_channels=3, 
                     classes=1,
                     encoder_weights=None)
        model = load_pytorch_model(model_path, model, ignore_suffix='model')
        model.eval()
        model.to(device)
        models.append(model)
        
    # inference
    img_info_dict = {}
    for i in range(len(target_df)):
        img_id = target_df.iloc[i]['id']
        img_info = get_img_info(img_id, conf.tile_size, conf.margin, train=True)
        img_info_dict[img_id]=img_info
        #print('predicting: {}'.format(img_id))
        pred_mask = inference(models, img_id, img_info, conf.tile_size, conf.margin, conf.reduce, train=True)
        #print('load gt mask: {}'.format(img_id))
        gt_mask = rle2mask(target_df.iloc[i]['encoding'], shape=(img_info['width'],img_info['height']))
        #print('calc dice: {}'.format(img_id))
        dice = ((gt_mask * pred_mask).sum()*2)/(gt_mask.sum() + pred_mask.sum())
        
        pred_rle = rle_encode_less_memory(pred_mask)
        target_df.iat[i, 1] = pred_rle
        target_df.iat[i, 3] = dice
        print(f'{img_id}, {dice}')
    #print(target_df)
    
    target_df.to_csv(os.path.join(conf.output, f'fold{conf.fold}_pred.csv'), index=False)
    
if __name__ == "__main__":
    main()
