"""
Created on Fri Jul  7 17:58:15 2023
@author: Hugo —— A silly dog crying in the late night.
% If you have any question, please feel free to contact with Baidu.
"""
import os.path
from Three_phase import Thr_phase_lbt, Thr_phase_sample
from SFDInet import Model
import numpy as np
import scipy.io as scio

def get_imlist(path):
    
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')], [f.split('.')[0] for f in os.listdir(path) if f.endswith('.bmp')]

def main():
    
    np.random.seed(0)
    ##DMD and CCD  systerm parameter.
    DG = np.array([[[12.6, 12.5, 12.5]]], dtype='float32')
    ##config path/window size/ROI and file_path.ad_ratio
    ad_ratio = 0.9   
    # Win_size0 = [1096,896,1352,1152] #图片中心256*256
    Win_size0 = [968,768,1480,1280] #图片中心512*512
    """
    
    Guys, don't forget to change the path.
    
    """ 
    ##Picture path
    lbt_path = r"G:\hugo\20220419\lbt0"
    sample_path= r"G:\notebook\test\Blood_glucose\Thr_pics\20220802"
    save_path= r"G:\Article\temp\new"
    
    ## get DC and AC of lbt by Three phase method
    AC0, DC0 = Thr_phase_lbt(lbt_path, Win_size0, DG)
    
    img_path0,img_name=get_imlist(sample_path)
    """
    Please notice this parameter which is extremely significant.
    index = 0:Demodulate every three images；index = 1：Demodulate three images in sequence
    """
    index = 0 
       
    if index == 0:    
        n_num = int(len(img_path0)/3.)
    else:
        n_num = len(img_path0)-2
    for i in range(n_num):
          
        if index == 0: 
            ##sample picture denoising
            img1 = img_path0[3*i+0]
            img2 = img_path0[3*i+1]
            img3 = img_path0[3*i+2]
        else:
            ##sample picture denoising
            if i == 0:
                img1 = img_path0[i+0]
                img2 = img_path0[i+1]
                img3 = img_path0[i+2]
            else:
                img1 = img2
                img2 = img3
                img3 = img_path0[i+2]
        ## get DC and AC of sample by Three phase method
        AC1, DC1 = Thr_phase_sample(img1, img2, img3, Win_size0, DG)
        ##MTF
        MTF_AC =  np.divide(AC1,(DC0*ad_ratio))
        MTF_DC =  np.divide(DC1,DC0)
        MTF0 = np.concatenate((MTF_DC, MTF_AC), axis = 2)
        ##extract optical properties and physiological parameters by CNN
        prediction = Model(MTF0)
        ##save result
        scio.savemat(os.path.join(save_path, img_name[i]+'.mat'), {'zz':prediction[0,:,:,:], 'MTF':MTF0})
    

if __name__ == '__main__':

    main()
    


