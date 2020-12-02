from argparse import ArgumentParser
import os
import pandas as pd

import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

 
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



def split_image_mask(image, mask, size):
    #Pay attention to memory usage
    channel, width, height = image.shape[2], image.shape[1],image.shape[0]
    
    # padding image and mask
    pad_w, pad_h = size - width%size, size - height%size
    image = np.pad(image,[[0,pad_h],[0,pad_w],[0,0]],constant_values=0)
    mask = np.pad(mask,[[0,pad_h],[0,pad_w]],constant_values=0)
    #print(image.shape)
    
    # split
    num_split_w, num_split_h = int(image.shape[1]/size), int(image.shape[0]/size)
    split_img = np.zeros((num_split_h, num_split_w, size, size, 3))
    split_mask = np.zeros((num_split_h, num_split_w, size, size))
    for h in range(num_split_h):
        for w in range(num_split_w):
            split_img[h,w,:] = image[h*size:(h+1)*size, w*size:(w+1)*size, :]
            split_mask[h,w,:] = mask[h*size:(h+1)*size, w*size:(w+1)*size]
    return split_img, split_mask



def split_save_image_mask(image, mask, size, save_dir, img_id):
    print('processing: {}'.format(img_id))
    #Pay attention to memory usage
    channel, width, height = image.shape[2], image.shape[1],image.shape[0]
    
    # padding image and mask
    print(channel, width, height)
    pad_w, pad_h = size - width%size, size - height%size
    print('pad_w: {}, pad_h: {}'.format(pad_w, pad_h))
    image = np.pad(image,[[0,pad_h],[0,pad_w],[0,0]],constant_values=0)
    mask = np.pad(mask,[[0,pad_h],[0,pad_w]],constant_values=0)
    #print(image.shape)
    
    # split
    num_split_w, num_split_h = int(image.shape[1]/size), int(image.shape[0]/size)
    for h in range(num_split_h):
        for w in range(num_split_w):
            if np.sum(image[h*size:(h+1)*size, w*size:(w+1)*size, :])!=0:
                Image.fromarray(image[h*size:(h+1)*size, w*size:(w+1)*size, :]).save(os.path.join(save_dir,'size_{}/w{}-h{}-{}_image.png'.format(size,w,h,img_id)))
                Image.fromarray(mask[h*size:(h+1)*size, w*size:(w+1)*size]).save(os.path.join(save_dir,'size_{}/w{}-h{}-{}_mask.png'.format(size,w,h,img_id)))
    return


def main(args):
    train = pd.read_csv(os.path.join(args.data_dir,'train.csv'))
    os.makedirs(os.path.join(args.save_dir,'size_'+str(args.size)), exist_ok=True)
    
    for idx, row in train.iterrows():
        #print(row['id'])
        #print(row['encoding'])
        img_name = os.path.join(args.data_dir, 'train/' + row['id'] + '.tiff')
        img = np.squeeze(tiff.imread(img_name))
        if(img.shape[0] == 3):
            img = np.transpose(img, (1,2,0))

        encoding = row['encoding']
        mask = rle2mask(encoding,(img.shape[1],img.shape[0]))

        split_save_image_mask(img, mask, args.size, args.save_dir, row['id'])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--size", help="image size", type=int, required=False, default=256,
    )
    parser.add_argument(
        "-sd", "--save_dir", help="path to save", type=str, required=True,
    )
    parser.add_argument(
        "-dd", "--data_dir", help="path to data dir", type=str, required=True,
    )

    # args = parser.parse_args(['-dd', '../input/prostate-cancer-grade-assessment/', '-sd','../working'])
    args = parser.parse_args()

    main(args)