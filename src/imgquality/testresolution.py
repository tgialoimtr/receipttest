'''
Created on Nov 7, 2017

@author: loitg
'''

import os
import cv2
import numpy as np
from classify.common import sharpen, sauvola, ASHOW
from skimage.filters import gaussian

receiptpath = '/home/loitg/Downloads/complex-bg/'

def sharpen(binimg, blur1, blur2, alpha):
    blurred_l= gaussian(binimg, blur1)
#     blurred_l= gaussian(binimg,0.8) CMND
    filter_blurred_l = gaussian(blurred_l, blur2)
#     filter_blurred_l = gaussian(blurred_l, 0.4)  # CMND
    return blurred_l + alpha * (blurred_l - filter_blurred_l)

def invespic(rawimg):
    def nothing(x):
        pass
    cv2.namedWindow('image')
    cv2.createTrackbar('blur1','image',0,10,nothing)
    cv2.createTrackbar('blur2','image',0,10,nothing)
    cv2.createTrackbar('alpha','image',2,50,nothing)
    
    while(1):
        blur1 = cv2.getTrackbarPos('blur1','image')/10.0
        blur2 = cv2.getTrackbarPos('blur2','image')/10.0
        alpha = cv2.getTrackbarPos('alpha','image')
        
        img = sharpen(rawimg, blur1, blur2, alpha)
        
#         img = gaussian(rawimg, blur1*2)
        
        
        bin = sauvola(img, w=31, k=0.2, reverse=True)*255
        bin = cv2.resize(bin, None, fx=3.0,fy=3.0)
        cv2.imshow('image',bin)
        
        bin2 = sauvola(rawimg, w=31, k=0.2, reverse=True)*255
        bin2 = cv2.resize(bin2, None, fx=3.0,fy=3.0)        
        cv2.imshow('raw', bin2)
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
        
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for filename in os.listdir(receiptpath):
        if filename[-3:].upper() != 'JPG' and filename[-3:].upper() != 'PNG': continue
        img = cv2.imread(receiptpath + filename,0)
        (h,w) = img.shape[:2]
        
        ah = min(h/2,360)
        aw = min(w/2,360)
        sample = img[h/2-ah/2:h/2+ah/2, w/2-aw/2:w/2+aw/2]
#         invespic(sample)
        
        ASHOW('sample', )
        
        
        
        
        
        
        
        