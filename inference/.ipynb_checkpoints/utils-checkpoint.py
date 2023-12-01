import os
import glob
import numpy as np
import nibabel as nib
import torch
import subprocess
import scipy.ndimage
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff

def orientation(img):
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = axcodes2ornt('RAS')
    transform = ornt_transform(orig_ornt, targ_ornt)
    return img.as_reoriented(transform)

def find_bbox(affine, fwd_path):
    MNI_coords = [95,128,64] #MNI coords for hypothalamus
    fwd_img = orientation(nib.load(fwd_path)).get_fdata()
    ras_coords = np.append(fwd_img[MNI_coords[0],MNI_coords[1],MNI_coords[2]], 1).reshape(-1,1) 
    hypo_coords = np.ceil(np.linalg.inv(affine)@ras_coords)
    pixdim_init = np.sqrt(np.sum(affine[:-1,:-1]**2, axis=0))
    D,H,W = int(hypo_coords[0]), int(hypo_coords[1]), int(hypo_coords[2])
    d,h,w = int(np.ceil(30/pixdim_init[0])), int(np.ceil(30/pixdim_init[1])), int(np.ceil(30/pixdim_init[2]))
    return (D,H,W), (d,h,w)