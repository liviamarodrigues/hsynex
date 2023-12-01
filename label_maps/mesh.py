import os
import glob
import nibabel as nib
import numpy as np
from utils_mirror import find_corners, mesh_sides

def mesh(initial_path, initial_save_path, margin, res):
    rotated_path = sorted(glob.glob(os.path.join(initial_path, '*')))
    for idx, path in enumerate(rotated_path):
        path_right = sorted(glob.glob((os.path.join(path, 'right*'))))
        path_left = sorted(glob.glob((os.path.join(path, 'left**'))))
        subject = path.split('/')[-1]
        print('working on subject:', subject)
        subj_path = os.path.join(initial_save_path, subject)
        if not os.path.isdir(subj_path):
            os.mkdir(subj_path)
        for idx_k in range(len(path_right)):
            value_k = path_right[idx_k].split('_')[-1]
            print(path_right[idx_k]) 
            save_path = os.path.join(subj_path, value_k)
            
            left_side = nib.load(path_left[idx_k])
            img_left = left_side.get_fdata()
            max_val = img_left.max()
            aff_left = left_side.affine
            k_ = int(value_k.split('.')[0])
            img_left[img_left>k_] = img_left[img_left>k_]+max_val #changing values from the segmented structure in one side
            corner1, corner2 = find_corners(img_left, aff_left, margin)
            left = mesh_sides(img_left, aff_left, corner1, corner2,res,True)
            print('LEFT done')
    
            right_side = nib.load(path_right[idx_k])
            img_right = right_side.get_fdata()
            aff_right = right_side.affine
            right = mesh_sides(img_right, aff_right, corner1, corner2,res,False)
            img_affine = np.array([[res,0,0,corner1[0]],[0,res,0,corner1[1]],[0,0,res,corner1[2]],[0,0,0,1]])
            print('RIGHT done')
    
            mirrored = np.zeros((left.shape[0]+right.shape[0], left.shape[1], left.shape[2]))
            mirrored[:left.shape[0]]=left
            mirrored[left.shape[0]:] =right
            nib.save(nib.Nifti1Image(mirrored, img_affine), save_path)
            print(f'DONE. Saved in {save_path}')
    
