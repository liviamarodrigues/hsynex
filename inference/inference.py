import os
import torch
import scipy.ndimage
import numpy as np
import torch.nn as nn
import torch.utils
import nibabel as nib
from utils import orientation, find_bbox
from pytorch3dunet.unet3d.model import UNet3D
from nibabel.orientations import ornt_transform

class InVivoInference():
    
    def __init__(self, device, voxres):

        self.voxres = voxres
        self.aff = np.eye(4)    
        self.device = device
        self.ckpt_hypo = 'https://github.com/liviamarodrigues/hsynex/releases/download/weights/all_hypo.ckpt'
        self.ckpt_subnuclei = 'https://github.com/liviamarodrigues/hsynex/releases/download/weights/subn_hypo.ckpt'
        
    def load_model(self, ckpt, model, idx):
        teste_hyp = torch.hub.load_state_dict_from_url(ckpt)
        tt_hyp = teste_hyp['state_dict']
        new_state_dict_seg_hyp = {}
        for keys_hyp, values_hyp in tt_hyp.items():
            if keys_hyp == 'CE.weight':
                pass
            else: 
                new_state_dict_seg_hyp[keys_hyp[idx:]] = values_hyp.float()  
        model.load_state_dict(new_state_dict_seg_hyp)
        model.float().to(self.device)
        model.eval()
        return model
    
    def rescale_pred(self, prediction, original_affine, original_shape):
        self.coords, self.gap = find_bbox(original_affine, self.fwd_path)
        D,H,W = self.coords
        d,h,w = self.gap
        D1,W1,H1 = self.img_shape[1], self.img_shape[2], self.img_shape[3]
        orig = nib.io_orientation(self.aff)
        targ = nib.io_orientation(original_affine)
        transform = ornt_transform(orig, targ)
        pred_nifti = nib.Nifti1Image(prediction, self.aff)
        pred_reor = pred_nifti.as_reoriented(transform).get_fdata()
        pred_HR = np.zeros((200,200,200))
        pred_HR[int(D1/2)-80:int(D1/2)+80, int(W1/2)-80:int(W1/2)+80, int(H1/2)-80:int(H1/2)+80] = pred_reor
        final_pred =np.zeros(original_shape)        
        pred_LR = scipy.ndimage.zoom(pred_HR, [2*d/200,2*h/200, 2*w/200], order =0, grid_mode = True, mode = 'grid-constant')
        final_pred[D-d:D+d, H-h:H+h, W-w:W+w] = pred_LR
        return final_pred

    def input_adjust(self, input_img, dilated_ss, vdc):
        img= orientation(input_img)
        self.aff = img.affine
        img_array = img.get_fdata()
        img_array_flip = img_array.copy()
        img_array_flip[0] = img_array[0, ::-1]
        self.img_shape = img_array.shape
        D,W,H = self.img_shape[1], self.img_shape[2], self.img_shape[3]
        img_array_flip = img_array_flip[:, int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]
        img_array = img_array[:, int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]        
        cropped_ss = dilated_ss[int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]     
        cropped_vdc = vdc[int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80]
        input_torch = torch.from_numpy(img_array.astype(np.float16)).view(1,self.img_shape[0],160,160,160).to(self.device)
        input_torch_flip = torch.from_numpy(img_array_flip.astype(np.float16)).view(1,self.img_shape[0],160,160,160).to(self.device)
        cropped_torch_ss = torch.from_numpy(cropped_ss.astype(np.float16)).view(1,1,160,160,160).to(self.device)
        return input_torch , input_torch_flip, cropped_torch_ss, cropped_vdc

    def find_logits(self, input, cropped_ss):    
        soft = nn.Softmax(dim=1) 
        unet_hypo = UNet3D(4,2, f_maps=(24,48,96,192,384),layer_order= 'gcr',num_groups =4, is_segmentation=False)
        unet_subnuclei = UNet3D(4,13, f_maps=(24,48,96,192,384),layer_order= 'gcr',num_groups =6, is_segmentation=False)
        model_hypo = self.load_model(self.ckpt_hypo, unet_hypo, 13)
        model_subnuclei = self.load_model(self.ckpt_subnuclei, unet_subnuclei, 6)
        with torch.no_grad():    
            pred_hypo = model_hypo(input.float())
        pred_hypo = soft(pred_hypo)[:,0].view(1,1,160,160,160)>0.5
        final_input = pred_hypo*input*cropped_ss
        with torch.no_grad():    
            pred_subnuclei = model_subnuclei(final_input.float())   
        return pred_subnuclei
    
    def pred_image(self, input_img, dilated_ss, vdc):     
        soft = nn.Softmax(dim=1)    
        right_labels = [1,2,3,4,5,6] #prediction right hypothalamus labels 
        left_labels = [7,8,9,10,11,12] #prediction left hypothalamus labels 
        order_flip = [0]+left_labels+right_labels
        input, input_flip,cropped_ss, cropped_vdc = self.input_adjust(input_img, dilated_ss, vdc)
        logit = self.find_logits(input, cropped_ss)
        logit_flip = self.find_logits(input_flip, cropped_ss.flip(dims=(2,)))
        prob_map = soft(logit)
        prob_map_flip = soft(torch.flip(logit_flip,[2])) #undoing the flip
        prob_map_flip_organized = prob_map_flip[:, order_flip,...] #adjusting channels 
        prob_average = (prob_map_flip_organized+prob_map)/2 #averaging both predictions
        final_segmentation = prob_average[0].argmax(axis = 0)       
        return final_segmentation.float().cpu().numpy()*cropped_vdc

    def save_nifti(self, seg_final, save_path_nib, original_affine, original_shape):
        if self.voxres == 'original':
            seg_final = self.rescale_pred(seg_final, original_affine, original_shape) #cropping VDC to [160,160,160]
            nib_img_itk = nib.Nifti1Image(seg_final, original_affine)
        else:
            seg_final = seg_final.float().cpu().numpy()
            D,W,H = self.img_shape[1], self.img_shape[2], self.img_shape[3]
            seg_final_save = np.zeros((D,W,H))
            seg_final_save[int(D/2)-80:int(D/2)+80, int(W/2)-80:int(W/2)+80, int(H/2)-80:int(H/2)+80] = seg_final
            seg_final_save = seg_final_save*dilated_ss
            nib_img_itk = nib.Nifti1Image(seg_final_save, self.aff)
        nib.save(nib_img_itk, save_path_nib)

    def predict(self, original_path, save_path, input_img, dilated_ss, vdc):
        tmp_path = os.path.join(save_path, 'tmp') 
        name = os.path.basename(original_path)        
        img_original = nib.load(original_path)
        self.fwd_path = os.path.join(tmp_path,'fwd_'+name) 
        self.ss_path = os.path.join(tmp_path,'ss_'+name)      ##   
        save_path_nib = os.path.join(save_path, 'hsynex_'+name)
        try:
            final_segmentation = self.pred_image(input_img, dilated_ss, vdc)
            print('prediction DONE')
            print('saving...')
            self.save_nifti(final_segmentation, save_path_nib, img_original.affine, img_original.shape)
            print('saved in', save_path_nib)
        except torch.cuda.OutOfMemoryError:
            raise MemoryError("CUDA out of memory. Try --device cpu")
        except:
            print("Error on:", name)
