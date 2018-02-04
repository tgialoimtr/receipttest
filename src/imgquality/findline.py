'''
Created on Nov 7, 2017

@author: loitg
'''

import os
import cv2
import numpy as np
from classify.common import sharpen, sauvola, ASHOW, summarize
from skimage.filters import gaussian
from scipy.ndimage import morphology,measurements,filters
from scipy import signal
from matplotlib import pyplot as plt

# from scipy.signal import stft

receiptpath = '/home/loitg/Downloads/complex-bg/'

def sharpen(binimg, blur1, blur2, alpha):
    blurred_l= gaussian(binimg, blur1)
#     blurred_l= gaussian(binimg,0.8) CMND
    filter_blurred_l = gaussian(blurred_l, blur2)
#     filter_blurred_l = gaussian(blurred_l, 0.4)  # CMND
    return blurred_l + alpha * (blurred_l - filter_blurred_l)
def sl_dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop-s[0].start
def sl_dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop-s[1].start
def sl_area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return np.prod([max(x.stop-x.start,0) for x in a[:2]])

def abc(binary):
    size = (2,2)
    binary = filters.maximum_filter(binary,size,origin=0)
    binary = filters.minimum_filter(binary,size,origin=0)
    labels,_ = measurements.label(binary)
    objects = measurements.find_objects(labels) ### <<<==== objects here
    bysize = sorted(range(len(objects)), key=lambda k: sl_area(objects[k]))
    scalemap = np.zeros(binary.shape)
    smalldot = np.zeros(binary.shape, dtype=binary.dtype)
    for i in bysize:
        o = objects[i]
        if np.amax(scalemap[o])>0: 
            continue
#         scalemap[o] = sl_area(o)**0.5
        scalemap[o] = 1.0
    boxmap = filters.minimum_filter(scalemap,(2,2),origin=0)
    scalemap = filters.gaussian_filter(1.0*scalemap,(0.1*15, 1*15),order=(0,0))
    cv2.imshow('scalemap', (scalemap*255).astype(np.uint8))  
    _,_, Sxx = signal.spectrogram(scalemap[:,0], 100, nperseg=120 ,noverlap=100)
    freqmap = np.zeros((Sxx.shape[1], binary.shape[1],3))
    _,_, Sxx = signal.spectrogram(boxmap[0,:], 100, nperseg=60 ,noverlap=50)
    boxfreq = np.zeros((binary.shape[0], Sxx.shape[1]))
    for i in range(1, binary.shape[0]):
        f,t, Sxx = signal.spectrogram(boxmap[i,:], 100, nperseg=60 ,noverlap=50)
        boxfreq[i,:] = np.amax(Sxx,axis=0)        
    for i in range(1, binary.shape[1]):
        f,t, Sxx = signal.spectrogram(scalemap[:,i], 100, nperseg=120 ,noverlap=100)
        freqmap[:,i,1] = np.argmax(Sxx[3:15,:],axis=0)
        freqmap[:,i,0] = np.amax(Sxx[3:15,:],axis=0)
#         plt.plot(scalemap[:,i])
#         plt.show()
#         plt.pcolormesh(t, f, Sxx)
#         plt.ylabel('Frequency [Hz]')
#         plt.xlabel('Time [sec]')
#         plt.show()
#     temp = cv2.normalize(freqmap[:,:,0], None, alpha=np.uint8(0), beta=np.uint8(255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     th, freqmap[:,:,0] = cv2.threshold(temp,0.0,1.0,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    freqmap[:,:,0] = cv2.normalize(freqmap[:,:,0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    freqmap[:,:,1] = cv2.normalize(freqmap[:,:,1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    boxfreq = cv2.normalize(boxfreq, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    boxfreq = cv2.resize(boxfreq, (binary.shape[1],binary.shape[0])  , None)
    freqmap = cv2.resize(freqmap, (binary.shape[1],binary.shape[0])  , None)
#     freqmap[:,:,0] = sauvola(freqmap[:,:,0], w=binary.shape[1]/3, scaledown=0.2)
    freqmap[:,:,2] = boxfreq
    cv2.imshow('freq',freqmap)
    cv2.waitKey(-1)

    
    
#     grad = gaussian_filter(1.0*cleaned,(0.3*scale, 6*scale),order=(1,0))
#     scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here

def horizentalSTFT(img, window_size):
    (h,w) = img.shape[:2]
    for x in xrange(img.shape[1]):
        for y in xrange(img.shape[0]):
            img.item(y,x)

if __name__ == '__main__':
    for filename in os.listdir(receiptpath):
        if filename[-3:].upper() != 'JPG' and filename[-3:].upper() != 'PNG': continue
        img = cv2.imread(receiptpath + filename,0)
        (h,w) = img.shape[:2]
        
        binary = sauvola(img, w=31, k=0.2, scaledown=0.5, reverse=True)
        cv2.imshow('debug', binary*255)
        
        abc(binary)

        
        
        
        
        
        
        
        