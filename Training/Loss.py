import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

#target is one_hot
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, y_pred, y_true,from_logits=True):
        if from_logits:
            BCE_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(y_pred, y_true, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class MacroF1Loss(nn.Module):
    def __init__(self):
        super(MacroF1Loss,self).__init__()
    def forward(self, y_pred,y_true,from_logits=True):
        if from_logits:
            y_pred=F.softmax(y_pred)
        tp = torch.sum(y_true*y_pred, dim=0)
        tn = torch.sum((1-y_true)*(1-y_pred), dim=0)
        fp = torch.sum((1-y_true)*y_pred, dim=0)
        fn = torch.sum(y_true*(1-y_pred), dim=0)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)

        f1 = 2*p*r / (p+r+1e-8)
        return 1 - torch.mean(f1)

#y_pred from logits,y_true is number
class CrossEntropyWithWeights(nn.Module):
    def __init__(self,weigths):
        super(CrossEntropyWithWeights,self).__init__()
        self.weights=weigths
    def forward(self, y_pred, y_true):
        weights=torch.zeros_like(y_true,dtype=torch.float32)
        for idx,label in enumerate(y_true):
            weights[idx]=self.weights[label]
        return torch.mean(weights*F.cross_entropy(y_pred,y_true,reduce=False))
if __name__=='__main__':
    batch_size=32
    x =torch.randn(batch_size,5)
    print(x)
    y=  torch.zeros((batch_size,5),dtype=torch.long)
    for i in range(batch_size):
        y[i][random.randint(0,4)]=1
    print(y)
    print(MacroF1Loss()(x,y))
        
