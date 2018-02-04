'''
Created on Nov 19, 2017

@author: loitg
'''
import sys
import logging
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter, uniform_filter
from time import time
import ocrolib
from ocrolib.toplevel import *
from ocrolib import psegutils,morph,sl
from skimage.filters import threshold_sauvola
from scipy.ndimage import morphology
# from classify.common import DSHOW

def summarize(a):
    b = a.ravel()
    return a.shape,a.dtype, [np.amin(b),np.mean(b),np.amax(b)], np.percentile(b, [0,20,40,60,80,100])
def ASHOW(title, image, scale=1.0, waitKey=False):
    HEIGHT = 800.0
    if len(image.shape) > 2:
        h,w,_ = image.shape
    else:
        h,w = image.shape
    canhlon = h if h > w else w
    tile = HEIGHT/canhlon
    
    mm = np.amax(image)
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
        
# allcmnd = '/home/loitg/Downloads/complex-bg/'
allcmnd = '/home/loitg/workspace/receipttest/img/'
erc1 = cv2.text.loadClassifierNM1('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM1.xml')
erc2 = cv2.text.loadClassifierNM2('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM2.xml')

angles = list(np.arange(-45,-5,10)) + list(np.arange(-5,5,1)) + list(np.arange(5,85,10)) + list(np.arange(85,95,2))
angles = dict(zip(range(len(angles)), angles))
def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res

def sauvola(grayimg, w=51, k=0.2, scaledown=None, reverse=False):
    mask =None
    if scaledown is not None:
        mask = cv2.resize(grayimg,None,fx=scaledown,fy=scaledown)
        w = int(w * scaledown)
        if w % 2 == 0: w += 1
        mask = threshold_sauvola(mask, w, k)
        mask = cv2.resize(mask,(grayimg.shape[1],grayimg.shape[0]),fx=scaledown,fy=scaledown)
    else:
        if w % 2 == 0: w += 1
        mask = threshold_sauvola(grayimg, w, k)
    if reverse:
        return np.where(grayimg > mask, np.uint8(0), np.uint8(1))
    else:
        return np.where(grayimg > mask, np.uint8(1), np.uint8(0))

def compute_boxmap(binary,scale,oriimg,threshold=(.5,2),dtype='i'):
    labels,n = morph.label(binary)
    objects = morph.find_objects(labels)
    boxmap = np.zeros(binary.shape,dtype)
    for i,o in enumerate(objects):
        h = sl.dim0(o)
        w = sl.dim1(o)
        ratio = float(h)/w if h > w else float(w)/h
        if h > 2*scale or h < scale/3:
            continue
        if ratio > 8: continue
        if sl.area(o)**.5<threshold[0]*scale: continue
        if sl.area(o)**.5>threshold[1]*scale: continue
        boxmap[o] = 1
    return boxmap

def compute_gradmaps(binary,scale):
    # use gradient filtering to find baselines
    binaryary = morph.r_opening(binary.astype(bool), (1,1))# CMND
    boxmap = compute_boxmap(binaryary,scale,binary)
    cleaned = boxmap*binaryary
    # this uses non-Gaussian oriented filters
    grad = gaussian_filter(1.0*cleaned,(max(4,0.3*scale), 1.0*scale),order=(1,0))
    grad = uniform_filter(grad,(1.0,1*scale)) # CMND
    bottom = ocrolib.norm_max((grad<0)*(-grad))
#     bottom = minimum_filter(bottom,(2,6*scale))
    top = ocrolib.norm_max((grad>0)*grad)
#     top = minimum_filter(top,(2,6*scale))
    return bottom,top,boxmap

def compute_line_seeds(binaryary,bottom,top,scale):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = 0.5
    vrange = int(scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked = bmarked*(bottom>t*np.amax(bottom)*t)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked = tmarked*(top>t*np.amax(top)*t/2)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = np.zeros(binaryary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
#     DSHOW("lineseeds",[0.4*seeds,0.3*tmarked+0.7*bmarked,binaryary])
    return seeds

@checks(SEGMENTATION)
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,sampling=[3,1], return_distances=1,return_indices=1) #CMND
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread
    
def localVarWithAngle(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (nW, nH),borderValue=1.0)
    
    temp = localVar(rotated)
    M = cv2.getRotationMatrix2D((rotated.shape[1]//2, rotated.shape[0]//2), angle, 1.0)
    M[0, 2] -= (nW / 2) - cX
    M[1, 2] -= (nH / 2) - cY
    return cv2.warpAffine(temp, M, (w,h))

def localVar(img):
    temp = uniform_filter(1.0*img,(1.5, 9*5))
    average = uniform_filter(temp,(2*5, 2*5)) - temp
    return uniform_filter(average*average,(5*5, 5*5))

def firstAnalyse(binaryary):
    labels,_ = morph.label(binaryary)
    objects = morph.find_objects(labels) ### <<<==== objects here
    bysize = sorted(objects,key=sl.area)
    scalemap = np.zeros(binaryary.shape)
    for o in bysize:
        if np.amax(scalemap[o])>0: continue
        scalemap[o] = sl.area(o)**0.5
    scale = np.median(scalemap[(scalemap>10)&(scalemap<40)]) ### <<<==== scale here
    return objects, scale

def findTextInImage(img):
    temp = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    channels = cv2.text.computeNMChannels(temp)
    channel = channels[0]
    er1 = cv2.text.createERFilterNM1(erc1,60,0.000015,0.0004,0.5,True,0.7)    
    regions = cv2.text.detectRegions(channel,er1,None)
    i=0
    
    if len(regions) < 2:
        return []
    rects = cv2.text.erGrouping(temp,channel,[r.tolist() for r in regions])
    rs = []
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        rs.append((slice(rect[1],rect[1]+rect[3],None),slice(rect[0],rect[0]+rect[2],None)))
    return rs

if __name__ == '__main__':
    rs = 0
    try:
        filename = sys.argv[1]
        print filename
        img0 = cv2.imread(allcmnd + filename,0)
        img = cv2.resize(img0, None, fx=0.25,fy=0.25)
        img = sauvola(img, w=11, k =0.1)
#         ASHOW('temp',img)
        allrotate = np.zeros((img.shape[0], img.shape[1], len(angles)))
        for i,a in angles.iteritems():       
            temp = localVarWithAngle(img, a)
            allrotate[:,:,i] = temp    
        allrotate = uniform_filter(allrotate, (5,5,2.5))
        textangle = np.argmax(allrotate, 2).astype(np.uint8)
        textangle = np.clip(textangle,0,len(angles)-1)
 
        textangle = np.vectorize(angles.get)(textangle)
        anglemask = cv2.inRange(textangle, -5,5)
#         anglemask = maximum_filter(anglemask, (2*5,5))
        temp = cv2.bitwise_and(textangle,textangle, mask=anglemask)
        h,w = temp.shape[:2]
        temp = temp[h/4:3*h/4,w/4:3*w/4]
        anglemask_center = anglemask[h/4:3*h/4,w/4:3*w/4]
        totalpixels = np.sum(anglemask_center)/255.0
        average_angle = np.sum(temp)/totalpixels
        
        density = totalpixels/(w*h/4)
        
        M = cv2.getRotationMatrix2D((img0.shape[1]//2, img0.shape[0]//2), -average_angle, 1.0)
        img0 = cv2.warpAffine(img0, M, (img0.shape[1],img0.shape[0]))
        anglemask = cv2.resize(anglemask, (img0.shape[1], img0.shape[0]), None)
        anglemask = cv2.warpAffine(anglemask, M, (img0.shape[1],img0.shape[0]))
        
   
        img_bin = sauvola(img0, scaledown=0.25, reverse=True)
        parragraph = cv2.bitwise_and(img_bin,img_bin, mask=anglemask)
        objects, scale = firstAnalyse(parragraph)
 
#         bottom,top,boxmap = compute_gradmaps(parragraph,scale)
#         seeds0 = compute_line_seeds(parragraph,bottom,top,scale)
#         seeds,_ = morph.label(seeds0)
#           llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
#           spread = spread_labels(seeds,maxdist=scale)
#           llabels = np.where(llabels>0,llabels,spread*parragraph)
#           segmentation = llabels*parragraph
#           parragraph = ocrolib.remove_noise(parragraph, 8)
#           lines = psegutils.compute_lines(segmentation,scale/2)
#           lines = [linedesc.bounds for linedesc in lines]

        img_bin = sauvola(img0, scaledown=0.25)
        parragraph = cv2.bitwise_and(img_bin,img_bin, mask=anglemask)    
        lines = findTextInImage(parragraph)
        
        bad_resolution = 0
        blur = 0
        both = 0
        total = 0
        for i,linedesc in enumerate(lines):
            y0,x0,y1,x1 = [int(x) for x in [linedesc[0].start,linedesc[1].start, \
              linedesc[0].stop,linedesc[1].stop]]     
            line = img0[y0:y1, x0:x1]
            temp = cv2.Laplacian(line, cv2.CV_64F).var()
            if  temp < 700:
                blur += 1
            if y1-y0 < 15:
                bad_resolution += 1
            if temp < 1000 and y1-y0 < 20:
                both += 1
            total += 1
        

        if density < 0.25:
            print 'Too inclined'
            rs = rs | (1<<0)
        elif total < 4:
            print 'Discontinuous character'
            rs = rs | (1<<1)
        else:
            if 1.0*bad_resolution/total > 0.6:
    #             print bad_resolution,'/',total
                print 'Low resolution line'
                rs = rs | (1<<2)
            if 1.0*blur/total > 0.6:
    #             print blur,'/',total
                print 'Blurred'
                rs = rs | (1<<3)
            if 1.0*both/total > 0.5:
    #             print both,'/',total
                print 'Blurred and small'
                rs = rs | (1<<4)
    except:
        logging.exception("Something awful happened!")
        print 'GOOD'
        rs = 0
    print bin(rs)   
    sys.exit(rs)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        