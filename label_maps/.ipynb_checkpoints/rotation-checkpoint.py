import os
import glob
import scipy.optimize
import numpy as np
import nibabel as nib
from utils_mirror import orientation, transform, cost_fun

def rotation(save_path, initial_path, alpha):
    merged_path = sorted(glob.glob(os.path.join(initial_path, '*')))
    for idx, path in enumerate(merged_path):
        subject = path.split('/')[-1]
        print('working on', subject) 
        print('using alpha =',alpha)
        subj_dir = os.path.join(save_path,subject)
        if not os.path.isdir(subj_dir):
            os.mkdir(subj_dir)
        path_subject = sorted(glob.glob(os.path.join(path,'**')))
        nib_img = nib.load(path_subject[0])
        aff = nib_img.affine
        img_ = nib_img.get_fdata()
        (W,L,C) = img_.shape
        transformed_img = np.zeros((W,L,C))
        params = [0,0,0,0,0,0]
        coord = np.argwhere(img_!=0)
        i, j, k = coord[:,0], coord[:,1], coord[:,2]
        ci, cj, ck =np.median(i), np.median(j), np.median(k)
        aux = aff@np.array([ci, cj, ck, 1]).reshape(4,1)
        cx, cy, cz = int(np.round(aux[0])), int(np.round(aux[1])), int(np.round(aux[2]))
        xopt = scipy.optimize.fmin(func=cost_fun, x0=params, args = (aff, cx, cy, cz, i, j, k, alpha), maxiter=500)
        print("Optimization DONE")
        RAS, aff_new = transform(xopt, aff, cx, cy, cz, i, j, k, alpha)
        aff_neg =aff_new.copy()
        aff_neg[0] = -aff_neg[0]
        for idx_k, path_k in enumerate(path_subject):
            name = os.path.basename(path_k)
            img_k = nib.load(path_k).get_fdata()
            left = nib.Nifti1Image(img_k, aff_new)
            right = nib.Nifti1Image(img_k, aff_neg)
            nib.save(left, subj_dir+'/left_'+name)
            nib.save(right, subj_dir+'/right_'+name)
            print(f'(image {name} Saved in {subj_dir}')
