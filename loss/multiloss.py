import torch
from torch import nn
import numpy as np
from catalyst.contrib.nn.criterion.dice import BCEDiceLoss

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.seg_criteria = BCEDiceLoss()
        self.cls_criteria = nn.BCEWithLogitsLoss()

    def forward(self, y_hat, y, y_label, label):
        seg_loss = self.seg_criteria(y_hat, y)
        cls_loss = self.cls_criteria(y_label, label)
        
        loss = cls_loss + seg_loss

        return loss

class OUSMMultiLoss(nn.Module):
    def __init__(self):
        super(OUSMMultiLoss, self).__init__()
        self.seg_criteria = BCEDiceLoss()
        self.cls_criteria = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_hat, y, y_label, label):
        bs = y_label.shape[0]
        losses = self.cls_criteria(y_label, label)
        if len(losses.shape) == 2:
            losses = losses.mean(1)
        _, idxs = losses.topk(int(bs*0.8), largest=False)
        
        cls_loss = losses.index_select(0, idxs).mean()
        
        seg_loss = self.seg_criteria(y_hat.index_select(0, idxs), y.index_select(0, idxs))
        
        loss = cls_loss + seg_loss

        return loss
