import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torchio as tio
import kornia.augmentation as K
from torch.utils.data import Dataset
from pytorch3dunet.unet3d.model import UNet3D
from skimage.morphology import closing, cube

model_path = '/space/calico/3/asterion/1/users/lr252/synth_hypothalamus/exvivo_synth/final_codes/training/all_hypo.ckpt'
teste = torch.load(model_path)
modelseg = UNet3D(4,2, f_maps=(24,48,96,192,384),layer_order= 'gcr',num_groups =4, is_segmentation=False)
tt = teste['state_dict']
new_state_dict_seg = {}
for keys, values in tt.items():
    new_state_dict_seg[keys[13:]] = values.float()     
modelseg.load_state_dict(new_state_dict_seg)

class ExVivoDataset(Dataset):
    
    def __init__(self, labels_list, device):
        self.labels_list = labels_list
        self.device = torch.device(device)
        self.model = modelseg.to(self.device)
        self.model = self.model.eval()
        
    def __len__(self):
        return 13*len(self.labels_list)  
    
    def augmentation(self):        
        elastic = K.RandomElasticTransform(kernel_size=(23,23), sigma=(8,8), alpha =(.5, .5), p = 1)
        distortion = K.RandomPerspective(distortion_scale=0.1)
        crop = K.RandomCrop3D((160,160,160))
        return nn.Sequential(elastic, distortion, crop)       

    def gmm_(self, synth_brain, k_):
        volume_factor = 1 / .3 / .3 / .3
        channels = synth_brain.shape[0]       
        hypo = np.zeros((160,160,160))
        annotation = np.zeros((12,160,160,160))
        anterior = [k_, k_+5, k_+6, k_+11] #labels for anterior subnuclei
        for idx in range(k_, k_+12):
            nuclei = synth_brain[idx].cpu().numpy()
            annotation[idx-k_] = nuclei
            if idx != k_+4 and idx != k_+10: #label for fornix, taking out of hypot. 
                hypo += nuclei
        hypo = (hypo>0).astype(int)
        hypo = torch.from_numpy(closing(hypo, cube(2))).to(self.device)
        synth_brain_final = synth_brain.clone()
        synt_brain_final = synth_brain[:k_+3] # Adding all k_ values plus 3 channels:
        synth_brain_final[-3] = 1-(synth_brain.sum(axis=0)>0).long() #background
        synth_brain_final[-2] = synth_brain[k_+4]+synth_brain[k_+10] #right and left fornix as one label
        synth_brain_final[-1] = hypo # whole hypo (once again, we can not use the subnuclei separately here)
        ch_=synth_brain_final.shape[0]
        mean =  torch.from_numpy(15 + 240 * np.random.rand(ch_,1,1,1)).to(self.device)
        stdev = torch.from_numpy( np.sqrt(volume_factor)*(10 + 25  * np.random.rand(ch_,1,1,1))).to(self.device)
        variances = stdev * stdev
        mean_image = (synth_brain_final * mean).sum(axis=0)
        var_image = (synth_brain_final * variances).sum(axis=0)
        std_image = torch.sqrt(var_image)
        gaus_image = torch.from_numpy(np.random.normal(size = synth_brain_final.shape[1:])).to(self.device)
        gmm = std_image*gaus_image + mean_image        
        gmm[gmm<0] = 0
        return gmm, torch.from_numpy(annotation).to(self.device)

    def final_transforms(self):
        bfield = tio.transforms.RandomBiasField(order=1)
        blur = tio.transforms.RandomBlur([1.2, 2.2])
        return tio.transforms.Compose([bfield, blur])
    
    def __getitem__(self, idx):
        idx_ = idx//13       
        labels_one = torch.from_numpy(np.load(self.labels_list[idx_])).to(self.device)
        k_ = labels_one.shape[0]-15 #find k_ labels from k-means
                                    #excluding 10 hypothalamus subnuclei, fornix and bak channels
    
        (C,D,H,W) = labels_one.shape
        labels_ = labels_one.view(1,C,D,H,W).to(self.device)        
        augmentation_transform = self.augmentation()
        affine = K.RandomAffine3D(degrees = (15,15,15), p = 1)
        labels_rotated = affine(labels_)
        transformed = augmentation_transform(labels_rotated[0].to(self.device))
        gmm_img, annotation_ = self.gmm_(transformed[0,3:], k_) #excluding MNI coords to synth image generation
        annotation = torch.zeros((13,160,160,160)).to(self.device)
        annotation[1:] = (annotation_>0).long().view(12,160,160,160)
        bkgrnd = (transformed[0,k_+3:]>0.9).long().sum(dim=0)
        annotation[0] = 1-(bkgrnd>0).long()
        blur = self.final_transforms()
        final_brain = blur(gmm_img.view(1,160,160,160).cpu()).to(self.device)
        bak = (transformed[0,:3]/100).view(3,160,160,160)
        final_brain = (final_brain - final_brain.min())/(final_brain.max()-final_brain.min()) 
        final_brain = final_brain.view(1,160,160,160)
        final_input = torch.cat((final_brain, bak), dim=0).view(1,4,160,160,160)
        with torch.no_grad():    
            pred_array = self.model(final_input.to(self.device).float())
        soft = nn.Softmax()
        pred = soft(pred_array)[:,0].view(1,1,160,160,160)>0.5
        final_input = pred*final_input
        return final_input[0], annotation