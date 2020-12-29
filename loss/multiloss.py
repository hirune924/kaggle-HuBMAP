import torch
from torch import nn
import numpy as np
import catalyst.contrib.nn.criterion.dice.BCEDiceLoss as BCEDiceLoss

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

