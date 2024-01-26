import os
import subprocess
import scipy.ndimage
import numpy as np
import nibabel as nib
from utils import orientation, find_bbox
from skimage.morphology import dilation, cube

class CreateInput():
    
    def __init__(self):
        self.atlas_path = 'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'
        self.atlas_seg_path = 'mni_icbm152_t1_tal_nlin_asym_09c.seg.nii.gz' 

    def create_tmp_files(self, path, save_path):        
        name = os.path.basename(path)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        self.tmp_path = os.path.join(save_path, 'tmp')
        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)        
        self.ss_path = os.path.join(self.tmp_path, 'ss_'+name)
        self.bak_path = os.path.join(self.tmp_path,'bak_'+name)
        self.fwd_path = os.path.join(self.tmp_path,'fwd_'+name)
        if not os.path.isfile(self.ss_path) or not os.path.isfile(self.bak_path) or not os.path.isfile(self.fwd_path):
            subprocess.run(['mri_easyreg', '--ref', self.atlas_path, '--flo', path, '--ref_seg', self.atlas_seg_path, '--flo_seg', self.ss_path, '--bak_field', self.bak_path, '--threads', '32', '--fwd_field', self.fwd_path])
            subprocess.run(['mri_convert', self.ss_path, self.ss_path, '-rl', path,'-rt', 'nearest', '-odt', 'float'])

    def adjust_ss_mask (self, D,H,W,d,h,w):
        elem = np.zeros((3,3,3))
        elem[:,2]=elem[...,2]=elem[2]=1
        exclude_labels = [15,14,24]  #excluding CSF and cerebelum
        synthseg_img = orientation(nib.load(self.ss_path)).get_fdata()
        for lbl in exclude_labels:
            synthseg_img[synthseg_img==lbl]=0        
        synthseg_img = synthseg_img[D-d:D+d, H-h:H+h, W-w:W+w]
        ventral_dc = (synthseg_img==28).astype(int) + (synthseg_img==60).astype(int)
        return dilation(synthseg_img>0, elem), dilation(ventral_dc>0, elem)
        
    def find_masked(self, original):        
        original_img = original.get_fdata() 
        bak_img = orientation(nib.load(self.bak_path)).get_fdata()     
        self.coords, self.gap = find_bbox(original.affine, self.fwd_path)
        D,H,W = self.coords
        d,h,w = self.gap
        cropped_img = original_img[D-d:D+d, H-h:H+h, W-w:W+w]
        dilated_ss, ventral_dc = self.adjust_ss_mask (D,H,W,d,h,w)        
        cropped_bak = bak_img[D-d:D+d, H-h:H+h, W-w:W+w,:]/100
        masked_img = dilated_ss*cropped_img        
        normalized_img = (masked_img - masked_img.min())/(masked_img.max()-masked_img.min())
        return normalized_img, cropped_bak, ventral_dc

    def adjust_affine(self, affine):        
        D,H,W = self.coords
        d,h,w = self.gap
        new_voxel_size = 0.3
        aff_cropped = affine.copy()
        aff_cropped[:-1, -1]=aff_cropped[:-1, -1]+aff_cropped[:-1, :-1]@np.array([D-d,H-h, W-w])
        pixdim = np.sqrt(np.sum(aff_cropped[:-1,:-1]**2, axis=0))
        aff_scaled = aff_cropped.copy()
        factor = pixdim/new_voxel_size
        aff_scaled[:3, :3] = aff_scaled[:3, :3]/factor
        aff_scaled[:-1, -1]=aff_scaled[:-1, -1]-aff_scaled[:-1, :-1]@((factor-1)/2)
        return aff_scaled, factor

    def extract_image(self, original_path, save_path):     
        name = os.path.basename(original_path)
        self.create_tmp_files(original_path, save_path)    
        print('SynthSeg DONE')
        original = orientation(nib.load(original_path))
        masked_img, cropped_bak, ventral_dc = self.find_masked(original)
        corrected_affine, factor = self.adjust_affine(original.affine)   
        rescaled_img = scipy.ndimage.zoom(masked_img, [factor[0], factor[1], factor[2]], order =1, grid_mode = True, mode = 'grid-constant')
        rescaled_ss = scipy.ndimage.zoom(ventral_dc, [factor[0], factor[1], factor[2]], order =0, grid_mode = True, mode = 'grid-constant')
        rescaled_bak = scipy.ndimage.zoom(cropped_bak, [factor[0], factor[1], factor[2],1], order =1, grid_mode = True,mode = 'grid-constant').transpose(3,0,1,2)  
        final_input = np.concatenate((rescaled_img[np.newaxis,:], rescaled_bak), axis=0)
        return nib.Nifti1Image(final_input, corrected_affine), rescaled_ss>0
