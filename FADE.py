# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:00:39 2020

"""

#%% load packages
import numpy as np
import copy
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import color

from scipy.io import loadmat, savemat
from scipy import ndimage


#  local package
from FADE_tools import *


#%% functions
def FADE(im, Id=None):
    """
    Obtain the perceptual fog density.
    This code is a python version of original MATLAB code.
    Please cite the original paper if you use the code.

    References
    --------------------
    L. K. Choi, J. You, and A. C. Bovik, "Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging",
    % IEEE Transactions on Image Processing, to appear (2015)
    https://live.ece.utexas.edu/research/fog/fade_defade.html


    Parameters
    ----------
    im (numpy array) : an image to be evaluated
    Id (numpy array) : the dark channel of input image. Default is None.

    Returns
    -------
    density (float) : a value to indicate the haze/fog density of the input image

    """

    #%% Pre-processing
    # change the type of input image
    I = im.astype('float')

    try:
        rows, cols, channels = I.shape
        print('Image shape:', rows, cols, channels)
    except Exception:
        print('Please input a 3D image!')

    # reshape the image, its size should be times of 8*8
    ps = 8

    # cut image to 8*n, n is an integer
    n_row, n_col = int(rows/ps), int(cols/ps)

    I = I[:n_row*ps, :n_col*ps, :]
    print('Image shape after cropped:', I.shape)

    # get the R, G, and B channels, convert the color image to grayscale
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    Ig = np.rint(0.2989 * R + 0.5870 * G + 0.1140 * B)

    # other features, such as dark channel, saturation ...
    if not Id:
        Id = np.amin(I, axis=2)/255  # dark channel

    I_hsv = color.rgb2hsv(I)
    Is = I_hsv[:,:,1]  # saturation

    rg = I[:,:,0] - I[:,:,1]  # rg
    by = 0.5*(I[:,:,0] + I[:,:,1]) - I[:,:,2]  # by

    # MSCN
    kernel = matlab_style_gauss2D((7,7), 7/6)
    mu = ndimage.convolve(Ig, kernel, mode='nearest')
    mu_sq = mu*mu
    sigma = np.sqrt(np.abs(ndimage.convolve(Ig*Ig, kernel, mode='nearest')-mu_sq))
    MSCN = (Ig-mu) /(sigma+1)
    cv = sigma / mu

    #%% Features for evaluating haze/fog density
    # f1
    idx_i, idx_j = get_im2col_indices(MSCN.shape, 8, 8, 0, 8)

    rows_new, cols_new = -1, 1

    # f1
    MSCN_var = np.nanvar(MSCN[idx_i, idx_j], axis=0, ddof=1).reshape(rows_new, cols_new, order='F')

    # f2, f3
    MSCN_pair_col = MSCN * np.vstack((MSCN[-1,:], MSCN[:-1,:]))
    MSCN_pair_col = MSCN_pair_col[idx_i, idx_j]
    pair_1 = copy.deepcopy(MSCN_pair_col)
    pair_2 = copy.deepcopy(MSCN_pair_col)
    pair_1[MSCN_pair_col>0] = np.nan
    pair_2[MSCN_pair_col<0] = np.nan
    MSCN_V_pair_L_var = np.nanvar(pair_1, axis=0, ddof=1).reshape(rows_new, cols_new, order='F')
    MSCN_V_pair_R_var = np.nanvar(pair_2, axis=0, ddof=1).reshape(rows_new, cols_new, order='F')



    # f4
    Mean_sigma = np.mean(sigma[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')

    # f5
    Mean_cv = np.mean(cv[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')

    # f6, f7, f8
    CE_gray, CE_by, CE_rg = CE(I)
    Mean_CE_gray = np.mean(CE_gray[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')
    Mean_CE_by = np.mean(CE_by[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')
    Mean_CE_rg = np.mean(CE_rg[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')

    # f9
    IE = get_entropy(Ig.astype('int')[idx_i, idx_j]).reshape(rows_new, cols_new, order='F')

    # f10
    Mean_Id = np.mean(Id[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')

    # f11
    Mean_Is = np.mean(Is[idx_i, idx_j], axis=0).reshape(rows_new, cols_new, order='F')

    # f12
    rg_mat = rg[idx_i, idx_j]
    by_mat = by[idx_i, idx_j]
    rg_mean, rg_std = rg_mat.mean(axis=0), rg_mat.std(axis=0, ddof=1)
    by_mean, by_std = by_mat.mean(axis=0), by_mat.std(axis=0, ddof=1)
    CF = np.sqrt(rg_std**2 + by_std**2) + 0.3*np.sqrt(rg_mean**2 + by_mean**2)
    CF = CF.reshape(rows_new, cols_new, order='F')

    feat = np.hstack((MSCN_var, MSCN_V_pair_R_var, MSCN_V_pair_L_var, Mean_sigma, Mean_cv, Mean_CE_gray, Mean_CE_by, Mean_CE_rg, IE, Mean_Id, Mean_Is, CF))
    feat = np.log(1+feat)

    #%% get the haze/fog density

    # 1 -- get the distance in the haze-free metric
    fogfree = loadmat('natural_fogfree_image_features_ps8.mat')
    cov_fogfree = fogfree['cov_fogfreeparam']
    mu_fogfree = fogfree['mu_fogfreeparam']

    mu_fog = feat
    cov_fog = np.nanvar(feat.T, axis=0, ddof=1, keepdims=True).T
    feature_size = feat.shape[1]
    mu_mat = np.tile(mu_fogfree, (feat.shape[0],1)) - mu_fog

    idx_cov_temp1 = np.cumsum(feature_size*np.ones((1,len(cov_fog)))).astype('int')
    cov_temp1 = np.zeros((1, np.max(idx_cov_temp1)))
    idx_cov_temp1 -= 1
    cov_temp1[0, idx_cov_temp1] = 1
    idx_cov_temp2 = np.cumsum(cov_temp1)-cov_temp1
    cov_temp2 = cov_fog[idx_cov_temp2.astype('int').reshape(-1)]
    cov_temp3 = np.tile(cov_temp2, (1,feature_size))
    cov_temp4 = np.tile(cov_fogfree, (len(cov_fog), 1))
    cov_mat = (cov_temp3 + cov_temp4)/2

    cov_cell = cov_mat.reshape(-1,feature_size, feature_size)

    distance_patch = np.zeros((mu_mat.shape[0],1))
    for i in range(mu_mat.shape[0]):
        temp = np.dot(mu_mat[i], np.linalg.pinv(cov_cell[i]))
        distance_patch[i] = np.sum(temp * mu_mat[i])

    Df = np.nanmean(np.sqrt(distance_patch))

    # 2 -- get the distance in the haze metric
    foggy = loadmat('natural_foggy_image_features_ps8.mat')
    cov_foggy = foggy['cov_foggyparam']
    mu_foggy = foggy['mu_foggyparam']
    mu_mat = np.tile(mu_foggy, (feat.shape[0],1)) - mu_fog
    cov_temp5 = np.tile(cov_foggy, (cov_fog.shape[0],1))
    cov_mat = (cov_temp3 + cov_temp5)/2
    cov_cell = cov_mat.reshape(-1,feature_size, feature_size)
    distance_patch = np.zeros((mu_mat.shape[0],1))
    for i in range(mu_mat.shape[0]):
        temp = np.dot(mu_mat[i], np.linalg.pinv(cov_cell[i]))
        distance_patch[i] = np.sum(temp * mu_mat[i])

    Dff = np.nanmean(np.sqrt(distance_patch))

    density = Df/Dff
    return density
