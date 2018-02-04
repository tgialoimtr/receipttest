'''
Created on Aug 22, 2017

@author: loitg
'''

import cv2
from pylab import *
from skimage.filters import threshold_sauvola
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import os
# from subprocess import call
import ocrolib

imgpath = '/home/loitg/workspace/receipttest/img/'

class object:
    pass


args = object()
args.escale = 1.0
args.debug = 1
args.lo = 5
args.hi = 90
args.zoom = 0.5
args.range = 20
args.perc = 80

def dshow(image,info):
    if args.debug<=0: return
    ion(); gray(); imshow(image); title(info); ginput(1,args.debug)
    
if __name__ == '__main__':
#     for filename in os.listdir(imgpath):
#         if filename[-3:] == 'JPG':
#             call(['ocropus-nlbin', imgpath+filename, '-n' ,'--threshold', '0.6', '--maxskew', '0', '-o', 'ocropus'])
#             call(['mv','ocropus/0001.bin.png', 'ocropus/'+ filename + '.bin.png'])
#             call(['mv','ocropus/0001.nrm.png', 'ocropus/'+ filename + '.nrm.png'])
        
    
    
    
#     for filename in os.listdir(imgpath):
    filename = '23a.JPG'
    raw = ocrolib.read_image_gray(imgpath + filename)
    image = raw-amin(raw)
    image /= amax(image)

    comment = ""
    m = interpolation.zoom(image,args.zoom)
    m = filters.percentile_filter(m,args.perc,size=(args.range,2))
    if args.debug>0: 
        ha = np.ones(m.shape)
        ha = ha * m
        print ha.shape
        print np.max(ha)
        print np.min(ha)
        cv2.imshow('temp', ha)
        cv2.waitKey(-1)
    m = filters.percentile_filter(m,args.perc,size=(2,args.range))
    if args.debug>0: 
        ha = np.ones(m.shape)
        ha = ha * m
        print ha.shape
        print np.max(ha)
        print np.min(ha)
        cv2.imshow('temp', ha)
        cv2.waitKey(-1)
    m = interpolation.zoom(m,1.0/args.zoom)
    
#     if args.debug>0: clf(); imshow(m,vmin=0,vmax=1); ginput(1,args.debug)
#     w,h = minimum(array(image.shape),array(m.shape))
#     flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
#     if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)
    
#     if est is not None:
#         print filename
#         est = est.astype(float)
#         est = est/255.0
#         est = np.clip(est, 0, 1.0)
#         e = 1.0
#         v = est-filters.gaussian_filter(est,e*20.0)
#         v = filters.gaussian_filter(v**2,e*20.0)**0.5
#         v = (v>0.3*np.amax(v))
#          
#         v = morphology.binary_dilation(v,structure=np.ones((int(e*50),1)))
#         v = morphology.binary_dilation(v,structure=np.ones((1,int(e*50))))
#         ran = est[v]
#         lo = stats.scoreatpercentile(ran.ravel(),args.lo)
#         print lo
#         hi = stats.scoreatpercentile(ran.ravel(),args.hi)
#         print hi
#         est -= lo
#         est /= (hi-lo)
#          
#         if args.debug>0: 
#             ha = np.ones(est.shape)
#             ha = ha * est
#             print ha.shape
#             print np.max(ha)
#             print np.min(ha)
#             cv2.imshow('temp', ha)
#             cv2.waitKey(-1)
    
    
    
    
    
#     img = cv2.imread(imgpath + filename, 0)
#     h = img.shape[0]
#     w = img.shape[1]
#     
# # <box top='771' left='9' width='388' height='578'>
# # <box top='778' left='22' width='47' height='101'>
#     
#     for row in range(771, 771+578, 101//2 ):
#         for col in range(9, 9+388, 47//2):
#             patch = img[row:row+101, col:col+47]
# #             patch = cv2.fastNlMeansDenoising(patch,h=10)
#         #     cv2.normalize(patch,patch,alpha=0,beta=254,norm_type=cv2.NORM_MINMAX)
#             
#             v = cv2.calcHist([patch], [0], None, [100], [0, 255])
#             
#             print v.reshape((1,100))
#             cv2.imshow('temp', patch)
#             plt.plot(v)
#             plt.show()
#             cv2.waitKey(-1)
    
#     <box top='1193' left='5' width='357' height='138'>
#     img = cv2.imread(imgpath + filename, 0)
#     img = img[1193:1193+138, 5:5+357]
#     for i in range(0,241,10):
#         hihi = (img >= i) & (img < i + 10)
#         ha = np.ones(hihi.shape)
#         ha = ha * hihi
#         print i
#         cv2.namedWindow('temp', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('temp', ha)
#         cv2.waitKey(-1)
