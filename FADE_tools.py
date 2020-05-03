# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:10:12 2020

original comments:
    % Input 
    % I: input image    
    % Output
    % Perceived Contrast Energy for gray, blue-yellow, red-green color channels

    % parameters    
    % In this code, I followed Groen's experiment [28]: fixed sigma, weight, t1, t2,and t3    
    % sigma: 0.16 degree =~ 3.25pixel, filter windows =~ 20 x 20 = (-3.25*3:1:3.25*3)
    % semisaturation: 0.1 
    % t1, t2, t3 follows the EmpiricalNoiseThreshold used in [28]
"""

import numpy as np
from scipy.signal import convolve2d

#https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# from cs231n assignment 2/ im2col.py
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  H,W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.tile(np.arange(field_height), field_width)
  #i0 = np.tile(i0, C)
  i1 = stride * np.tile(np.arange(out_height), int(out_width))
  j0 = np.repeat(np.arange(field_width), field_height)
  j1 = stride * np.repeat(np.arange(out_width), int(out_height))
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1) 

  return (i.astype('int'), j.astype('int'))

def border_in(I, p):
    
    # rows
    upcp = I[:p, :]
    downcp = I[-p:,:]    
    Igtemp1 = np.vstack((upcp, I, downcp))
    
    # columns
    leftcp = Igtemp1[:, :p]
    rightcp = Igtemp1[:,-p:]
    
    return np.hstack((leftcp, Igtemp1, rightcp))

# https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def CE_elements(im, p, Gx, semisaturation, t):
    CE_out = np.zeros(im.shape)
    
    gray_temp1 = border_in(im, p)
    Cx_gray = conv2(gray_temp1, Gx, mode='same')
    Cy_gray = conv2(gray_temp1, Gx.T, mode='same')
    C_gray_temp2 = np.sqrt(Cx_gray**2 + Cy_gray**2)
    C_gray = C_gray_temp2[p:-p,p:-p]
    R_gray = C_gray*C_gray.max()/(C_gray + C_gray.max()*semisaturation)
    R_gray_temp1 = R_gray - t
    idx1 = R_gray_temp1 > 0.0000001
    CE_out[idx1] = R_gray_temp1[idx1]
    return CE_out
    

def CE(I):
        
    #Basic parameters
    sigma               = 3.25
    semisaturation      = 0.1
    t1                  = 9.225496406318721e-004 *255; #0.2353
    t2                  = 8.969246659629488e-004 *255; #0.2287
    t3                  = 2.069284034165411e-004 *255; #0.0528
    p            = 10
    
    
    # Gaussian & LoG & Retification, Normalization(?)
    break_off_sigma     = 3
    filtersize          = break_off_sigma*sigma
    x                   = np.arange(-filtersize, filtersize)
    Gauss               = 1/(np.sqrt(2 * np.pi) * sigma)* np.exp((x**2)/(-2 * sigma * sigma) )
    Gauss               = Gauss/(np.sum(Gauss))
    Gx                  = (x**2/sigma**4-1/sigma**2)*Gauss  #LoG
    Gx                  = Gx-np.sum(Gx)/len(x)
    Gx                  = Gx/np.sum(0.5*x*x*Gx).reshape(-1,1)
    
    I = I.astype('float')
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    gray = 0.299*R + 0.587*G + 0.114*B
    by   = 0.5*R + 0.5*G - B
    rg   = R - G
    
    
    CE_gray = CE_elements(gray, p, Gx, semisaturation, t1)
    CE_by = CE_elements(by, p, Gx, semisaturation, t2)
    CE_rg = CE_elements(rg, p, Gx, semisaturation, t3)
    
    return CE_gray, CE_by, CE_rg
 

def get_entropy(mat):
    """
    

    Parameters
    ----------
    mat : TYPE
        DESCRIPTION.

    Returns
    -------
    entropy of each column

    """
    num_row, num = mat.shape
    ent = np.zeros((1, num))
    for i in range(num):
        _, count = np.unique(mat[:,i], return_counts=True, axis=0)
        prob = count/num_row 
        ent[0,i] = np.sum((-1)*prob*np.log2(prob))
        
    return ent
    
