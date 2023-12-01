import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import  DataLoader
from label2img import ExVivoDataset
from lightning_model import MyModelLightning
from dataloader_val import ValDataset
device = torch.device('cuda:0')
import time

params = {"labels_list": '/space/calico/3/asterion/1/users/lr252/exvivo/original_images_03/network_input_10sub_newAnterior_MNIcerto/I45/4*', 
          "batch size": 1,
          "path_val":'/space/calico/3/asterion/1/users/lr252/invivo/doug_imgs/*/cropped/dilated_M*nii*.gz',
          "path_labels":'/space/calico/3/asterion/1/users/lr252/invivo/doug_imgs/*/cropped/MNI_label_subnuclei_newAnterior.nii.gz',
          "accumulation": 32,
          "lr": 5e-5,
          "max_epochs": 2000,
          "patience": 200,
          "min_delta": 0.001,
          "num_workers": 0,
          "exp_name": 'new_lr',
          "lbl2img_device": "cpu",
          "checkpoint_path": '/space/calico/3/asterion/1/users/lr252/synth_hypothalamus/models/last.ckpt',
          "gamma_dice" : 0.7,
          "gamma_ce" : 0.3,
          "weights_ce":  [0.04, 0.10,0.06,0.06,0.06,0.10,0.10,0.10,0.06,0.06,0.06,0.10,0.10],
         }

list_val_ = sorted(glob.glob(params['path_val']))
list_val = [sub for idx, sub in enumerate(list_val_) if os.path.basename(sub).split('_')[-1] != 'T2s.nii.gz' and os.path.basename(sub).split('_')[-1] != 'label.nii.gz']
list_labels = sorted(glob.glob(params['path_labels']))
labels_list = sorted(glob.glob(params['labels_list']))

train_dataset = ExVivoDataset(labels_list=labels_list, device = params["lbl2img_device"])
train_dataloader = DataLoader(train_dataset, batch_size = params["batch size"], shuffle = True, 
                              num_workers = params["num_workers"])
val_dataset = ValDataset(list_val=list_val, list_labels =  list_labels, device = params["lbl2img_device"])
val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False)

max_epochs = params['max_epochs']
accumulate_grad_batches = params['accumulation']

checkpoint_path = params['checkpoint_path']
checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
print(f'Files in {checkpoint_dir}: {os.listdir(checkpoint_dir)}')
print(f'Saving checkpoints to {checkpoint_dir}')
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                      filename = params['exp_name']+"-{epoch}-{val_dice:.2f}",
                                      monitor="val_dice",
                                           mode="max",
                                           save_last = True,
                                           save_top_k=15)

early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_dice", min_delta=params["min_delta"], patience=params["patience"], verbose=True, mode="max")

resume_from_checkpoint = None
if os.path.exists(checkpoint_path):
    print(f'Restoring checkpoint: {checkpoint_path}')
    resume_from_checkpoint = checkpoint_path

trainer = pl.Trainer(accelerator='gpu',
                    devices=1,
                     max_epochs=max_epochs,
                     accumulate_grad_batches=accumulate_grad_batches,                     
                     callbacks=[checkpoint_callback]
                    )


model = MyModelLightning(train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         lr = params["lr"],
                         gamma1 = params["gamma_ce"],
                         gamma2 = params["gamma_dice"],
                         weights = params["weights_ce"],
                         )

trainer.fit(model)
