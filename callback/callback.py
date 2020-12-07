from pytorch_lightning.callbacks import Callback
from hydra.utils import to_absolute_path
from utils.preprocessing import *
import shutil
import pandas as pd

class ValidWholeImageCallback(Callback):
    def __init__(self, target_id, cfg):
        super().__init__()
        self.target_id = target_id
        self.cfg = cfg
        self.img_info = {}
        self.scale_factor = 1024/cfg.image_size
        self.df = pd.read_csv(to_absolute_path(os.path.join(cfg.data_root, 'train.csv')))
        for i in target_id:
            print('Preprocessing: {}'.format(i))
            img_pth = to_absolute_path(cfg.data_root) + '/train/' + i + '.tiff'
            self.img_info[i] = inf_preprocess(img_pth, tile_size=3072, margin=512, img_id=i)

    def on_validation_epoch_end(self, trainer, pl_module):
        print('validating for wsi')
        dice = 0.0
        for k in self.img_info.keys():
            pred_mask = inference(pl_module.model, self.img_info[k], tile_size=3072, margin=512, scale_factor=self.scale_factor)
            
            dice += dice_fn(pred_mask, rle2mask(self.df[self.df['id']==k]['encoding'].iloc[-1], shape=(self.img_info[k]['width'], self.img_info[k]['height'])))/len(self.img_info.keys())
            print(dice)
        pl_module.log('val_dice_whole',dice)

    def on_fit_end(self, trainer, pl_module):
        shutil.rmtree('../tiles')
