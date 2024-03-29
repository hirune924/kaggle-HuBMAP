import pytorch_lightning as pl
from loss.loss import get_loss
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler

import torch
import numpy as np
from pytorch_lightning.metrics import Accuracy
import segmentation_models_pytorch as smp

from utils.utils import load_obj
import albumentations as A
from utils.preprocessing import *
import shutil



class LitClassifier(pl.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.criteria = get_loss(hparams.training.loss)
        #self.accuracy = Accuracy()
        self.dice =  smp.utils.losses.DiceLoss(activation='sigmoid')

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.hparams.training.optimizer)

        scheduler = get_scheduler(optimizer, self.hparams.training.scheduler)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.hparams.dataset.mixup:
            num_batch = self.hparams.dataset.batch_size
            alpha = 0.2
            #rnd = torch.from_numpy(np.random.beta(alpha,alpha,int(num_batch/2))).type_as(x)
            #rnd = rnd.reshape(int(num_batch/2), 1, 1, 1)
            #x = x[:int(num_batch/2)]*rnd + x[int(num_batch/2):]*(1-rnd)
            #y = y[:int(num_batch/2)]*rnd + y[int(num_batch/2):]*(1-rnd)
            rnd = torch.from_numpy(np.random.beta(alpha,alpha,1)).type_as(x)
            x = x[:int(num_batch/2)]*rnd + x[int(num_batch/2):]*(1-rnd)
        y_hat = self.model(x)
        if self.hparams.dataset.mixup:
            loss = self.criteria(y_hat, y[:int(num_batch/2)])*rnd + self.criteria(y_hat, y[int(num_batch/2):])*(1-rnd)
        else:
            loss = self.criteria(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        dice = 1-self.dice(y_hat, y)

        #self.log('val_loss', loss)
        #self.log('val_dice', dice)

        return {
            "val_loss": loss,
            "val_dice": dice
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_dice = torch.stack([x["val_dice"] for x in outputs]).mean()

        self.log('val_loss', avg_val_loss)
        self.log('val_dice', avg_val_dice)
        #y = torch.cat([x["y"] for x in outputs]).cpu()
        #y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        #preds = np.argmax(y_hat, axis=1)

        #val_accuracy = self.accuracy(y, preds)

        #self.log('avg_val_loss', avg_val_loss)
        #self.log('val_acc', val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        self.log('test_loss', loss)

    
