'''
Created on Oct 3, 2017

@author: loitg
'''

from pylab import *
import cv2
from scipy.ndimage import interpolation
from skimage.filters import threshold_sauvola, gaussian
from ocrolib import psegutils,morph,sl

cmnd_path = '/home/loitg/Downloads/cmnd_data/'
hoadon_path = '/home/loitg/workspace/python/python/img/'
tmp_path = '/tmp/loitg/'

class obj:
    def __init__(self):
        pass
args = obj()
args.binmaskscale = 0.4
args.heavyprocesscale = 0.4
args.deskewscale = 0.1
args.range = 10

args.zoom = 0.5
args.range = 20
args.debug = 1
args.perc= 80
args.escale = 1.0
args.threshold = 0.5
args.lo = 5
args.hi = 90
args.usegauss = False
args.vscale = 1.0
args.hscale = 1.0
args.threshold = 0.2
args.pad = 0
args.expand = 3
args.model = '/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz'
args.connect = 4
args.noise = 8


def summarize(a):
    b = a.ravel()
    return a.shape, [amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])

def DSHOW(title,image):
    if not args.debug: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
    if args.debug>0: imshow(image); ginput(timeout=-1)
    
def ASHOW(title, image, scale=1.0, waitKey=False):
    HEIGHT = 1000*0.6
    if len(image.shape) > 2:
        h,w,_ = image.shape
    else:
        h,w = image.shape
    canhlon = h if h > w else w
    tile = HEIGHT/canhlon
    
    mm = amax(image)
    if mm > 0:
        temp = image.astype(float)/mm
    else:
        temp = image.astype(float)
    
#     if len(image.shape) > 2:
#         temp = cv2.resize(temp,None,fx=tile*scale,fy=tile*scale)
#     else:
#         temp = interpolation.zoom(temp, tile*scale)
    temp = cv2.resize(temp,None,fx=tile*scale,fy=tile*scale)
    cv2.imshow(title, temp)
    if waitKey:
        cv2.waitKey(-1)

def sharpen(binimg):
    blurred_l= gaussian(binimg,2)  
    filter_blurred_l = gaussian(blurred_l, 1)  
    alpha = 30
    return blurred_l + alpha * (blurred_l - filter_blurred_l) 
    
def estimate_skew_angle(image,angles):
    estimates = []
    binimage = sauvola(image, 11, 0.1).astype(float)
    for a in angles:
        rotM = cv2.getRotationMatrix2D((binimage.shape[1]/2,binimage.shape[0]/2),a,1)
        rotated = cv2.warpAffine(binimage,rotM,(binimage.shape[1],binimage.shape[0]))
        v = mean(rotated,axis=1)
        d = [abs(v[i] - v[i-1]) for i in range(1,len(v))]
        d = var(d)
        estimates.append((d,a))
#     if args.debug>0:
#         plot([y for x,y in estimates],[x for x,y in estimates])
#         ginput(1,args.debug)
    _,a = max(estimates)
    return a

def sauvola(grayimg, w=51, k=0.2, scaledown=None, reverse=False):
    mask =None
    if scaledown is not None:
        mask = cv2.resize(grayimg,None,fx=scaledown,fy=scaledown)
        w = int(w * scaledown)
        if w % 2 == 0: w += 1
        mask = threshold_sauvola(mask, w, k)
        mask = cv2.resize(mask,(grayimg.shape[1],grayimg.shape[0]),fx=scaledown,fy=scaledown)
    else:
        mask = threshold_sauvola(grayimg, w, k)
    if reverse:
        return where(grayimg > mask, 0, 1)
    else:
        return where(grayimg > mask, 1, 0)

def firstAnalyse(binaryary):
    labels,_ = morph.label(binaryary)
    objects = morph.find_objects(labels) ### <<<==== objects here
    bysize = sorted(objects,key=sl.area)
    scalemap = zeros(binaryary.shape)
    for o in bysize:
        if amax(scalemap[o])>0: continue
        scalemap[o] = sl.area(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here
    return objects, scale

class MyClass(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        