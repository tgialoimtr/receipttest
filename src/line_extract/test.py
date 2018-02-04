'''
Created on Aug 28, 2017

@author: loitg
'''

from pylab import *
from numpy.ctypeslib import ndpointer
from numpy import percentile, amax, amin
import argparse,os,os.path
from scipy.ndimage import filters,interpolation,morphology,measurements, uniform_filter1d, maximum_filter1d, minimum_filter1d
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, minimum_filter
from scipy import stats
import multiprocessing
import ocrolib
from skimage.filters import threshold_sauvola
from time import time
from numpy.fft import fft, fftshift

from skimage.transform import radon

from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import cv2


# imgpath = '/home/loitg/workspace/receipttest/img/'
imgpath = '/home/loitg/workspace/python/python/img/'

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
args.connect = 4
args.noise = 8


def summarize(a):
    b = a.ravel()
    return len(b),[amin(b),mean(b),amax(b)], percentile(b, [0,20,40,60,80,100])
    

def pre_check_line(line):
    project = mean(1-line, axis=0)
    project = uniform_filter1d(project, line.shape[0]/3)
    m = mean(project)
    if (m > 0.13) & (1.0*line.shape[1]/line.shape[0] > 1.7):
        return True
    else:
        return False
        
class Predictor:
    def __init__(self):
        self.network = ocrolib.load_object(args.model,verbose=1)
        for x in self.network.walk(): x.postLoad()
        for x in self.network.walk():
            if isinstance(x,lstm.LSTM):
                x.allocate(5000)
        self.lnorm = getattr(self.network,"lnorm",None)
        pass

    def predict(self, line):
        temp = amax(line)-line
        temp = temp*1.0/amax(temp) 
        self.lnorm.measure(temp)
        line = self.lnorm.normalize(line,cval=amax(line))
#         cv2.imshow('temp', line)
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
        tempbox = binary[o]
        ave = mean(tempbox)
        ratio = float(h)/w if h > w else float(w)/h
        if ratio > 4: continue
        
        if sl.area(o)**.5<threshold[0]*scale: continue
        if sl.area(o)**.5>threshold[1]*scale: continue

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
                                            args.hscale*0.5*scale),order=(1,0))
        grad = uniform_filter(grad,(args.vscale,args.hscale*6*scale))
    bottom = ocrolib.norm_max((grad<0)*(-grad))
#     bottom = minimum_filter(bottom,(2,6*scale))
    top = ocrolib.norm_max((grad>0)*grad)
#     top = minimum_filter(top,(2,6*scale))
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

if __name__ == '__main__':
#     filename = 'aea.JPG'
    linepredictor = Predictor()
    cls_file = open('/home/loitg/workspace/receipttest/img/line_decision.csv', 'w')
    cls_file.write('mean, max_fft, mean_fft, dodeu, cycle, ratio, height, importance\n')
    for filename in os.listdir(imgpath):
        if filename[-3:] != 'jpg':
            continue
        image = ocrolib.read_image_gray(imgpath + filename)
        tt=time()
        m = interpolation.zoom(image,args.zoom)
        m = filters.percentile_filter(m,args.perc,size=(args.range,2))
        m = filters.percentile_filter(m,args.perc,size=(2,args.range))
        m = interpolation.zoom(m,1.0/args.zoom)
#         if args.debug>0: imshow(m,vmin=0,vmax=1,cmap='gray'); ginput(timeout=-1)
        w,h = minimum(array(image.shape),array(m.shape))
        flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
#         if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1,cmap='gray'); ginput(timeout=-1)
 
        print 'flatten ', time() - tt
        tt=time()
        temp = interpolation.zoom(flat,0.2)
#             if args.debug>0: clf(); imshow(temp,vmin=0,vmax=1,cmap='gray'); ginput(1,-1)
        fourier = fftshift(fft2(temp))
        sh = log(1 + abs(fourier))
        peak = amax(sh)
#             if args.debug>0: clf(); imshow(sh,vmin=0,vmax=peak,cmap='gray'); ginput(1,-1)
#             continue
          
        h, w = sh.shape
        direction = np.empty(sh.shape)
        ncount = 0.0
        nsum = 0.0
        for i in range(h):
            for j in range(w):
                ii = i - h/2
                jj = j - w/2
                if (abs(ii) < 6) & (abs(jj) < 6): continue
                if sh[i,j] > 0.4* peak:
                    direction = arctan2(-ii,jj)/3.14159*180.0
                    if direction < 0:
                        direction += 180.0
                    if (direction < 100) & (direction > 80):
                        ncount += abs(fourier[i,j])
                        nsum += direction*abs(fourier[i,j])
        if ncount != 0:
            newangle = (90.0 - nsum/ncount)*float(h)/w
            flat = interpolation.rotate(flat, newangle, cval=1.0)
#         if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1,cmap='gray'); ginput(1,-1)
        
        mask = threshold_sauvola(flat, 9, 0.04)
        binary = where(flat > mask, 0, 1)     
 
        print 'deskew and binarize ', time() - tt
        tt=time()         
        
#         binary = where(image > 0.5, 0, 1)
        
        binaryary = morph.r_closing(binary.astype(bool), (args.connect,1))
        labels,n = morph.label(binaryary)
        objects = morph.find_objects(labels) ### <<<==== objects here
        bysize = sorted(objects,key=sl.area)
        scalemap = zeros(binaryary.shape)
        for o in bysize:
            if amax(scalemap[o])>0: continue
            scalemap[o] = sl.area(o)**0.5
        scale = median(scalemap[(scalemap>3)&(scalemap<100)]) ### <<<==== scale here
#         print 'scale: ', scale
        
        bottom,top,boxmap = compute_gradmaps(binary,scale)
#         DSHOW('hihi', [0.5*bottom+0.5*top, boxmap, binary])
#         print 'boxmap,top,bottom ', time() - tt
        tt=time()

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
        
        print 'compute line ', time() - tt
        tt=time()
        continue
        location_text = []
        for i,l in enumerate(lines):
            binline = psegutils.extract_masked(1-binary,l,pad=args.pad,expand=args.expand)
            if pre_check_line(binline):
                pred = linepredictor.predict(binline)
                result = psegutils.record(bounds = l.bounds, text=pred)
                location_text.append(result)
            print filename, str(i)
#             cv2.waitKey(999999)

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
            
        for i, result in enumerate(location_text): 
            if result is not None:   
                print result.bounds[0].stop, result.text
        
        
#         lobjects = morph.find_objects(segmentation)
#         seedbound = zeros(binary.shape)
#           
#         for i,o in enumerate(lobjects):
#             seedbound[o] += 1
# #             seedbound[o[0].start:o[0].stop-1,o[1].start] = 1
# #             seedbound[o[0].start:o[0].stop-1,o[1].stop-1] = 1
# #             seedbound[o[0].start,o[1].start:o[1].stop-1] = 1
# #             seedbound[o[0].stop-1,o[1].start:o[1].stop-1] = 1
#         seedbound = seedbound / amax(seedbound)
#  
#         print 'rest ', time() - tt
#         tt=time()
                     
#         DSHOW("seedbound",[seeds0, seedbound, binary])
        
        
        
        
        
    