# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
import math 

##clean RGB
def clean_RGB(I):
      Mtx = np.array([[91.9768, 33.4529, 15.6019],            # R incident
                      [27.4841, 88.8429, 35.8042],            # G incident
                      [14.0702, 19.1002, 102.666]])-12.5      # B incident
                      
      Mtx = np.divide(Mtx, np.sum(Mtx, axis=1).reshape(3,1))
      Inew = np.dot(np.linalg.pinv(Mtx.T), I.reshape(I.shape[0]*I.shape[1], 3).T).T.reshape(I.shape[0], I.shape[1], 3)
      return Inew

##get the path of picture
def get_imlist(path):
    
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')], [f.split('.')[0] for f in os.listdir(path) if f.endswith('.bmp')]

## get Incident light intensity by Three phase shift method.
def Thr_phase_lbt(path, Win_size, DG):
    img_n, _=get_imlist(path)
    img_shape = (np.array(Image.open(img_n[0]), dtype='float32')).shape
    img_array1 = np.zeros(img_shape, dtype='float32')
    img_array2 = np.zeros(img_shape, dtype='float32')
    img_array3 = np.zeros(img_shape, dtype='float32')
    for i in range(0,int(len(img_n)/3.)):
        img1 = Image.open(img_n[3*i])
        img2 = Image.open(img_n[3*i+1])
        img3 = Image.open(img_n[3*i+2])
        img_array1 += np.array(img1, dtype='float32')
        img_array2 += np.array(img2, dtype='float32')
        img_array3 += np.array(img3, dtype='float32')
    A = img_array1[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]/(len(img_n)/3.)-DG
    B = img_array2[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]/(len(img_n)/3.)-DG
    C = img_array3[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]/(len(img_n)/3.)-DG
    A = clean_RGB(A)
    B = clean_RGB(B)
    C = clean_RGB(C)
    DC = np.zeros(A.shape, dtype='float32')
    AC = np.zeros(A.shape, dtype='float32')     
    AC[:,:,:] = ((np.sqrt(2.0)/3.0)*np.sqrt(np.square(A-B)+np.square(B-C)+np.square(C-A))).astype('float32')
    dc =(A+B+C)/3
    # DC[:,:,:] = (dc*math.pi/0.985).astype('float32')
    DC[:,:,:] = (dc*math.pi/0.4925).astype('float32')
    return AC, DC

def Thr_phase_sample(img1, img2, img3, Win_size, DG):
    img1 = np.array(img1, dtype='float32')
    img2 = np.array(img2, dtype='float32')
    img3 = np.array(img3, dtype='float32')
    A = img1[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]-DG
    B = img2[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]-DG
    C = img3[Win_size[1]:Win_size[3], Win_size[0]:Win_size[2], :]-DG
    A = clean_RGB(A)
    B = clean_RGB(B)
    C = clean_RGB(C)
    AC = ((np.sqrt(2.0)/3.0)*np.sqrt(np.square(A-B)+np.square(B-C)+np.square(C-A))).astype('float32')
    DC = ((A+B+C)/3).astype('float32')
    return AC, DC
