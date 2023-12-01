import os
import glob
import pdb
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import nibabel as nib
import pytorch_lightning as pl
from pytorch3dunet.unet3d.model import UNet3D
from torch.utils.data import Dataset, DataLoader
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff

model_path = '/space/calico/3/asterion/1/users/lr252/synth_hypothalamus/exvivo_synth/final_codes/training/all_hypo.ckpt'
teste = torch.load(model_path)
modelseg = UNet3D(4,2, f_maps=(24,48,96,192,384),layer_order= 'gcr',num_groups =4, is_segmentation=False)
tt = teste['state_dict']
new_state_dict_seg = {}
for keys, values in tt.items():
    new_state_dict_seg[keys[13:]] = values.float()     
modelseg.load_state_dict(new_state_dict_seg)

class ValDataset(Dataset):    
    def __init__(self, list_val, list_labels, device):
        self.list_val = list_val
        self.list_labels = list_labels
        self.device = torch.device(device)
        self.model = modelseg.to(self.device)
        self.model = self.model.eval()
        
    def __len__(self):
        return len(self.list_val)
    
    def orientation(self,img):
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = axcodes2ornt('RAS')
        transform = ornt_transform(orig_ornt, targ_ornt)
        return img.as_reoriented(transform)
    
    def __getitem__(self, idx): 
        path = self.list_val[idx]
        path_label = self.list_labels[idx]
        img_teste= self.orientation(nib.load(path)).get_fdata()
        labels = self.orientation(nib.load(path_label)).get_fdata()
        (Ch,D,W,H) = img_teste.shape
        img_teste = img_teste[:, int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]
        img_teste = torch.from_numpy(img_teste).view(1,4,160,160,160).to(self.device)
        with torch.no_grad():    
            pred_array = self.model(img_teste.float())
        soft = nn.Softmax()
        pred = soft(pred_array)[:,0].view(1,1,160,160,160)>0.5
        final_input = pred*img_teste
        labels = labels[int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]
        return (final_input[0], torch.from_numpy(labels))
            