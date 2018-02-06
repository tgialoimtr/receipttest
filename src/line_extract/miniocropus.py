#!/usr/bin/env python

import cv2
from pylab import *
from scipy.ndimage import morphology
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, uniform_filter1d
import ocrolib
from skimage.filters import threshold_sauvola

from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import time

import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
       
class obj:
    def __init__(self):
        pass
    

args = obj()
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
args.pad = 3
args.expand = 3
args.model = '/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz'
args.inputdir = '/root/ocrapp/tmp/cleanResult/'
args.connect = 4
args.noise = 8


def summarize(a):
    b = a.ravel()
    return len(b),[amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])
   
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
        return where(grayimg > mask, uint8(0), uint8(1))
    else:
        return where(grayimg > mask, uint8(1), uint8(0)) 

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

erc1 = cv2.text.loadClassifierNM1('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM1.xml')
erc2 = cv2.text.loadClassifierNM2('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM2.xml')
def findTextInImage(img):
    
    tt = time.time()
    channels = cv2.text.computeNMChannels(img)
    print 'channel, ', time.time()-tt

    
#     cv2.Laplacian(temp, cv2.CV_64F).var()
    vis = img.copy()
    lines = []
    for i,channel in enumerate(channels):
        er1 = cv2.text.createERFilterNM1(erc1,60,0.000015,0.00013,0.5,True,0.1)
        er2 = cv2.text.createERFilterNM2(erc2,0.5)
        
        regions = cv2.text.detectRegions(channel,er1,er2)
        
        i=0
        for points in regions:
            i +=1
            cv2.fillConvexPoly(vis, points, (255*(i%3), 255*((i+1)%3), 255*((i+2)%3)))
        
        if len(regions) < 2:
            continue
        rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
#         rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifier_erGrouping.xml',0.2)
        #Visualization
        for r in range(0, shape(rects)[0]):
            rect = rects[r]
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
            if rect[2] > 15 and rect[3] > 15:
                lines.append(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :])
                
    DSHOW("lineseeds",vis)
    return vis


def pre_check_line(line):
    project = mean(1-line, axis=0)
    project = uniform_filter1d(project, line.shape[0]/3)
    m = mean(project)
    if (m > 0.13) & (1.0*line.shape[1]/line.shape[0] > 1.7):
        return True
    else:
        return False

class TensorFlowPredictor:
    def __init__(self):
        pass

    def predict_async(self, image, action):
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image))
        request.inputs['width'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image.shape[1], shape=[1,1]))
        return 'tensorflow'  
     
class Predictor:
    def __init__(self):
        self.network = ocrolib.load_object(args.model,verbose=1)
        for x in self.network.walk(): x.postLoad()
        for x in self.network.walk():
            if isinstance(x,lstm.LSTM):
                x.allocate(5000)
        self.lnorm = getattr(self.network,"lnorm",None)

    def predict(self, line):
        temp = amax(line)-line
        temp = temp*1.0/amax(temp)
        self.lnorm.measure(temp)
        line = self.lnorm.normalize(line,cval=amax(line))
        line = lstm.prepare_line(line,args.pad)
        pred = self.network.predictString(line)
        pred = ocrolib.normalize_text(pred)
        return pred


def DSHOW(title,image):
    if not args.debug: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
    if args.debug>0: imshow(image); ginput(timeout=-1)
    
def compute_boxmap(binary,scale,threshold=(.5,4),dtype='i'):
    objects = psegutils.binary_objects(binary)
    boxmap = zeros(binary.shape,dtype)
    for o in objects:
        h = sl.dim0(o)
        w = sl.dim1(o)
        a = h*w
#         black = float(sum(binary[o]))/a
#         if sl.area(o)**.5<threshold[0]*scale: continue
#         if sl.area(o)**.5>threshold[1]*scale: continue
        if h > 5*scale: continue
#         if h < 0.4*scale: continue
        if w > 4*scale: continue
        if a < 0.25*scale*scale: continue
        ratio = float(h)/w if h > w else float(w)/h
        if ratio > 4: continue
#         if ratio < 2 and black > 0.8: continue
        boxmap[o] = 1
    return boxmap

def compute_gradmaps(binary,scale):
    # use gradient filtering to find baselines
    binaryary = morph.r_closing(binary.astype(bool), (args.connect,1))
    boxmap = compute_boxmap(binaryary,scale)
    cleaned = boxmap*binaryary
    if args.usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(args.vscale*0.3*scale,
                                            args.hscale*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,args.vscale*0.3*scale),
                                            args.hscale*0.5*scale),order=(1,0))   # <====== originally 1
        grad = uniform_filter(grad,(args.vscale,args.hscale*1*scale))   # <====== originally 6/35
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    #bottom = minimum_filter(bottom,(0,10*scale))
    top = ocrolib.norm_max((grad>0)*grad)
    #top = minimum_filter(top,(0,10*scale))
    return bottom,top,boxmap

def compute_line_seeds(binaryary,bottom,top,scale):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = args.threshold
    vrange = int(args.vscale*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked = bmarked*(bottom>t*amax(bottom)*t)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked = tmarked*(top>t*amax(top)*t/2)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = zeros(binaryary.shape,'i')
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
    DSHOW("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binaryary])
    return seeds


@checks(SEGMENTATION)
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,sampling=[3,1], return_distances=1,return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

@checks(ARANK(2),True,pad=int,expand=int,_=GRAYSCALE)
def extract_line(image,linedesc,pad=5):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0,x0,y1,x1 = [int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start, \
                  linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
    y0,x0,y1,x1 = (y0-pad,x0-pad,y1+pad,x1+pad)
    h,w = image.shape
    y0 = clip(y0,0,h)
    y1 = clip(y1,0,h)
    x0 = clip(x0,0,w)
    x1 = clip(x1,0,w)
    if y0 < y1 and x0 < x1:
        return image[y0:y1, x0:x1]
    else:
        return None

class PagePredictor:
    def __init__(self):
#         self.lock = threading.Lock()
        self.linepredictor = TensorFlowPredictor()
    def ocrImage(self, imgpath):
        tt=time.time()
        
        img_grey = ocrolib.read_image_gray(imgpath)
        (h, w) = img_grey.shape[:2]
        img00 = cv2.resize(img_grey[h/4:3*h/4,w/4:3*w/4],None,fx=0.5,fy=0.5)
#             cv2.imshow('debug', img00)
#             cv2.waitKey(-1)
        angle = estimate_skew_angle(img00,linspace(-5,5,42))
        print 'goc', angle
    
        rotM = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        img_grey = cv2.warpAffine(img_grey,rotM,(w,h))
#         cv2.imshow('debug', img_grey)
#         cv2.waitKey(-1)
        
        h,w = img_grey.shape
        img_grey = cv2.normalize(img_grey.astype(float32), None, 0.0, 0.99, cv2.NORM_MINMAX)
        binary = sauvola(img_grey, w=128, k=0.2, scaledown=0.2, reverse=True)
    
        binaryary = morph.r_closing(binary[h/4:3*h/4,w/4:3*w/4].astype(bool), (args.connect,1))
        labels,n = morph.label(binaryary)
        objects = morph.find_objects(labels) ### <<<==== objects here
        bysize = sorted(objects,key=sl.area)
        scalemap = zeros(binaryary.shape)
        for o in bysize:
            if amax(scalemap[o])>0: continue
            scalemap[o] = sl.area(o)**0.5
        scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here
        
        bottom,top,boxmap = compute_gradmaps(binary,scale)
#         DSHOW('hihi', [bottom, top, binary])
        
        print 'boxmap,top,bottom ', time.time() - tt
        tt=time.time()
#         img_color = (img_grey*255).astype(np.uint8)
#         img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
#         findTextInImage(img_color)
#         
#         print 'scenetext ', time.time() - tt
#         tt=time.time()
        
    #         DSHOW("bottom-top-boxmap",[bottom,top,boxmap])
    #         if args.debug>0: clf(); imshow(binary,vmin=0,vmax=1,cmap='gray'); ginput(1,-1) 
        seeds0 = compute_line_seeds(binary,bottom,top,scale)
        seeds,_ = morph.label(seeds0)
        
        llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
        spread = spread_labels(seeds,maxdist=scale)
        llabels = where(llabels>0,llabels,spread*binary)
        segmentation = llabels*binary     
        binary = ocrolib.remove_noise(binary,args.noise)
        lines = psegutils.compute_lines(segmentation,scale)
        
        print 'compute line ', time.time() - tt
        tt=time.time()
    
    
        location_text = []
        for i,l in enumerate(lines):
            binline = extract_line(img_grey,l,pad=args.pad)
            if binline is None: continue
            def action(pred):
                result = psegutils.record(bounds = l.bounds, text=pred)
                location_text.append(result)               
            self.linepredictor.predict_async(binline,action)


        for i, result in enumerate(location_text):
            if True: #len(result.text) < 8 and ('.' in result.text or '$' in result.text):
                linemap = []
                for j, insertedline in enumerate(location_text):
                    if (result.bounds[1].start > insertedline.bounds[1].stop):
                        value = abs(result.bounds[0].stop - insertedline.bounds[0].stop)
                        if value < 2*scale:
                            linemap.append((value, insertedline))
                if len(linemap) > 0:
                    j, preline = min(linemap)
                    preline.text += (' ' + result.text)
                    yy = slice(minimum(preline.bounds[0].start, result.bounds[0].start), maximum(preline.bounds[0].stop, result.bounds[0].stop))
                    xx = slice(minimum(preline.bounds[1].start, result.bounds[1].start), maximum(preline.bounds[1].stop, result.bounds[1].stop))
                    preline.bounds = (yy,xx)
                    result = None
            
        ret = ''
        for i, result in enumerate(location_text): 
            if result is not None:   
                ret += normalize_text(pred) + '\n'
    #             ocrolib.write_text(args.outtext+str(i)+".txt",pred)/home/loitg/Downloads/complex-bg
        return ret
                    
                    
if __name__ == "__main__":
    sys.argv = ['','/home/loitg/Downloads/complex-bg/1507607007955_49c4f2d9-b85d-43f6-b1e0-72c288d2af4d.JPG', '/home/loitg/temp.txt']
    for filename in os.listdir('/home/loitg/Downloads/complex-bg/'):
        if filename[-3:].upper() == 'JPG':
            ret = PagePredictor().ocrImage('/home/loitg/Downloads/complex-bg/' + filename)
            with open(sys.argv[2], 'w') as outputfile:
                outputfile.write(ret)        
        
        
    