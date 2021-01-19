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

import torch.nn as nn, torch.nn.functional as F
from copy import deepcopy

class EMA:
    """
    The module for exponential moving average
    """

    def __init__(self, model, ema_model, decay=0.999):
        self.decay = decay
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        # some of the quantity in batch norm is LongTensor,
        # we have to make them become float, or it will cause the
        # type error in mul_ or add_ in self.step()
        for p in ema_model.parameters():
            p.detach_()
        for i in range(len(self.ema_params)):
            self.ema_params[i] = self.ema_params[i].float()

    def step(self):
        # average all the paramters, including the running mean and
        # running std in batchnormalization
        for param, ema_param in zip(self.params, self.ema_params):
            # if param.dtype == torch.float32:
            ema_param.mul_(self.decay)
            ema_param.add_(param * (1 - self.decay))
            # if param.dtype == torch.float32:
            #     param.mul_(1 - 4e-5)
            
def robust_binary_crossentropy(pred, tgt, eps=1e-6):
    inv_tgt = 1.0 - tgt
    inv_pred = 1.0 - pred + eps
    return -(tgt * torch.log(pred + eps) + inv_tgt * torch.log(inv_pred))

class LitClassifier(pl.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.s_model = model
        self.t_model = deepcopy(model)
        self.criteria = get_loss(hparams.training.loss)
        #self.accuracy = Accuracy()
        self.dice =  smp.utils.losses.DiceLoss(activation='sigmoid')

    def on_train_start(self):
        # model will put in GPU before this function
        # so we initiate EMA and WeightDecayModule here
        self.ema = EMA(self.s_model, self.t_model, 0.999)
        # self.wdm = WeightDecayModule(self.classifier, self.hparams.weight_decay, ["bn", "bias"])

    def on_train_batch_end(self, *args, **kwargs):
        # self.ema.update(self.classifier)
        # wd = self.hparams.weight_decay * self.hparams.learning_rate
        # customized_weight_decay(self.classifier, self.hparams.weight_decay, ["bn", "bias"])
        # self.wdm.decay()
        self.ema.step()

    def forward(self, x):
        # use forward for inference/predictions
        return self.t_model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.s_model.parameters(), self.hparams.training.optimizer)

        scheduler = get_scheduler(optimizer, self.hparams.training.scheduler)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        #print(batch)
        labeled_batch, unlabeled_batch = batch
        x, y = labeled_batch
        un_x = unlabeled_batch
        #print(x.shape)
        #print(y.shape)
        #print(un_x.shape)
        #if self.hparams.dataset.cutmix:
        
        # supervised phase
        lam = np.random.beta(0.5, 0.5)
        rand_index = torch.randperm(x.size()[0]).type_as(x).long()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        y[:, :, bbx1:bbx2, bby1:bby2] = y[rand_index, :, bbx1:bbx2, bby1:bby2]
            
        y_hat = self.s_model(x)
        sup_loss = self.criteria(y_hat, y)
        
        # un-supervised phase
        lam = np.random.beta(0.5, 0.5)
        rand_index = torch.randperm(un_x.size()[0]).type_as(un_x).long()
        bbx1, bby1, bbx2, bby2 = rand_bbox(un_x.size(), lam)
        
        mixed_un_x = un_x.clone()
        mixed_un_x[:, :, bbx1:bbx2, bby1:bby2] = un_x[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        with torch.no_grad():
            logits_u0_t = self.t_model(un_x).detach()
            logits_u1_t = self.t_model(un_x[rand_index,:]).detach()
            logits_unsup_t = logits_u0_t
            logits_unsup_t[:, :, bbx1:bbx2, bby1:bby2] = logits_u1_t[:, :, bbx1:bbx2, bby1:bby2]
            
        logits_unsup_s = self.s_model(mixed_un_x)
        # Logits -> probs
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        conf_thresh = 0.97
        conf_per_pixel=False
        if conf_thresh > 0.0:
            # Compute confidence of teacher predictions
            conf_tea = prob_unsup_t.max(dim=1)[0]
            # Compute confidence mask
            conf_mask = (conf_tea >= conf_thresh).float()[:, None, :, :]
            # Record rate for reporting
            #conf_rate_acc += float(conf_mask.mean())
            # Average confidence mask if requested
            if not conf_per_pixel:
                conf_mask = conf_mask.mean()
            #loss_mask = loss_mask * conf_mask
            loss_mask = conf_mask
            
        consistency_loss = robust_binary_crossentropy(prob_unsup_s, prob_unsup_t)
        consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
        consistency_loss = (consistency_loss * loss_mask).mean()
        
        self.log('train_loss', sup_loss + consistency_loss, on_epoch=True)
        return sup_loss + consistency_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.t_model(x)
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
        y_hat = self.t_model(x)
        loss = self.criteria(y_hat, y)
        self.log('test_loss', loss)
        
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

    
