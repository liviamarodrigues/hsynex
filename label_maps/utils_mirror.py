import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from nibabel.orientations import axcodes2ornt, ornt_transform

def find_corners(img, aff, margin):
    coord = np.argwhere(img>0)
    coord_h=np.concatenate((coord, np.ones((coord.shape[0],1))),axis=1).T
    RAS = aff@coord_h
    minRAS = RAS.min(axis=1)[:-1]
    maxRAS = RAS.max(axis=1)[:-1]
    corner1 = minRAS-margin
    corner2 = np.array([-minRAS[0], maxRAS[1], maxRAS[2]])+margin
    return corner1, corner2

def mesh_sides(img,aff, corner1, corner2, res, esquerda):
    size= np.ceil((corner2-corner1)/res).astype(int)
    image_buffer=np.zeros((size))
    img_affine = np.array([[res,0,0,corner1[0]],[0,res,0,corner1[1]],[0,0,res,corner1[2]],[0,0,0,1]])
    idx = np.floor(-(corner1[0]/res)).astype(int)
    if esquerda:
        iv = np.arange(idx+1)
    else:
        iv = np.arange(idx+1,image_buffer.shape[0])
    jv = np.arange(image_buffer.shape[1])
    kv = np.arange(image_buffer.shape[2])
    i, j, k = np.meshgrid(iv,jv,kv, indexing='ij')

    irav = i.ravel()
    jrav = j.ravel()
    krav = k.ravel()
    o=np.ones(irav.shape)
    ijk=np.stack([irav, jrav, krav,o])
    ijk = ((np.linalg.inv(aff)@img_affine)@ijk).astype(int)
    i = ijk[0,:]
    j = ijk[1,:]
    k = ijk[2,:]
    i[i<0]=0
    j[j<0]=0
    k[k<0]=0
    i[i>(img.shape[0]-1)] = (img.shape[0]-1)
    j[j>(img.shape[1]-1)] = (img.shape[1]-1)
    k[k>(img.shape[2]-1)] = (img.shape[2]-1)
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    z = np.arange(img.shape[2])
    fn = RegularGridInterpolator((x,y,z), img)
    interp = fn( np.array([i,j,k]).T)
    try:
        final = interp.reshape((idx+1, image_buffer.shape[1], image_buffer.shape[2]))
    except:
        final = interp.reshape((idx, image_buffer.shape[1], image_buffer.shape[2]))
    return final

def cost_fun(params, aff, cx, cy, cz, i, j, k, alpha):
    RAS, _ = transform(params, aff, cx, cy, cz, i, j, k, alpha)
    x = RAS[0,:]
    cost = x[x>0].sum() - alpha * x.sum()
    return cost

def transform(params, aff, cx, cy, cz, i, j, k, alpha):
    rotx = params[0]/180 * np.pi
    roty = params[1]/180 * np.pi
    rotz = params[2]/180 * np.pi
    tx = params[3]
    ty = params[4]
    tz = params[5]

    Ttrans = np.array([[1, 0, 0, -cx],[0, 1, 0, -cy],[0, 0 ,1, -cz],[0, 0, 0, 1]])
    TrotX = np.array([[1, 0, 0, 0],[0, np.cos(rotx), -np.sin(rotx), 0],[0, np.sin(rotx), np.cos(rotx), 0],[0, 0, 0, 1]])
    TrotY = np.array([[np.cos(roty), 0, np.sin(roty), 0],[0, 1, 0, 0],[-np.sin(roty), 0, np.cos(roty), 0],[0, 0, 0, 1]])
    TrotZ = np.array([[np.cos(rotz), -np.sin(rotz), 0, 0],[np.sin(rotz), np.cos(rotz), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    Ttrans2 = np.array([[1, 0, 0, cx + tx],[0, 1, 0, cy+ty],[0 ,0, 1, cz+tz],[0, 0, 0, 1]])
    T = Ttrans2@TrotZ@TrotY@TrotX@Ttrans #TshearX@TshearY@TshearZ
    aff_new = T@aff
    mtx = np.concatenate((i.reshape(1,-1), j.reshape(1,-1), k.reshape(1,-1), np.ones((1,len(i)))), axis=0)
    RAS = aff_new@mtx
    return RAS, aff_new


def orientation(img):
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = axcodes2ornt('RAS')
    transform = ornt_transform(orig_ornt, targ_ornt)
    return img.as_reoriented(transform)
