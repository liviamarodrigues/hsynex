import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch3dunet.unet3d.model import UNet3D
device = torch.device("cuda:0")


class MyModelLightning(pl.LightningModule):

    def __init__(self, train_dataloader, val_dataloader, lr,  weights, gamma1, gamma2):
                 
        super(MyModelLightning, self).__init__()

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.lr = lr
        self.soft = nn.Softmax()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weights = weights
        self.model = UNet3D(4,13, f_maps=(24,48,96,192,384),layer_order= 'gcr',num_groups =6, is_segmentation=False).float().to(device)
        self.model.to(device)
        
    def dice_coeff(self, pred, target, epsilon=1e-6):
        pred = pred.float()
        target = target.float() 
        dice = 0
        batch_dice = 0
        w = [0.04, 0.10,0.06,0.06,0.06,0.10,0.10,0.10,0.06,0.06,0.06,0.10,0.10]  
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                inter = (pred[i,j]*target[i,j]).sum()
                sets_sum = (pred[i,j]+target[i,j]).sum()
                if sets_sum.item() == 0:
                    sets_sum = 2 * inter
                dice_ = ((2 * inter + epsilon) / (sets_sum + epsilon))*w[j]
                dice += 1-dice_
            batch_dice += dice/pred.shape[1]        
        return batch_dice / pred.shape[0]
    
    def find_dice(self, y_pred, y_true):
        return torch.sum(y_pred*y_true)*2.0 / ((torch.sum(y_pred) + torch.sum(y_true))+0.00001)
        
    def dice_val(self, pred, target):
        labels_pred = [3,9,4,10,1,7,6,12,2,8]
        labels_target = [3,9,4,10,1,7,6,11,2,8] 
        dice_=0
        batch_dice = 0
        for bs in range(pred.shape[0]):
            for lbl in range(len(labels_pred)):
                dice_ += self.find_dice(pred[bs]==labels_pred[lbl],target[bs]==labels_target[lbl])
            batch_dice += dice_/len(labels_pred)        
        return batch_dice / pred.shape[0]
        
    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, nbatch):    
        class_weights = torch.FloatTensor(self.weights).to(device)
        CE = nn.CrossEntropyLoss(class_weights)
        x,y = batch   
        bs = x.shape[0]       
        soft = nn.Softmax()
        logit = self(x.float())
        dice_loss = self.dice_coeff(soft(logit).to(device), y.to(device))
        ce_loss = CE(logit.to(device), y.to(device))
        final_loss = self.gamma1*ce_loss+self.gamma2*dice_loss
        return final_loss
    
    def validation_step(self, batch, nbatch):
        x,y = batch        
        soft = nn.Softmax()
        logit = self(x.float())
        seg_val = soft(logit).argmax(axis=1)
        print(seg_val.max())
        val_dice = self.dice_val(seg_val, y)
        self.log("val_dice", val_dice)
        return val_dice
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr
            )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self._train_dataloader
    def val_dataloader(self):
        return self._val_dataloader

