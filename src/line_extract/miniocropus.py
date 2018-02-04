#!/usr/bin/env python

from pylab import *
# from numpy.ctypeslib import ndpointer
# import argparse,os,os.path
from scipy.ndimage import morphology
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter, uniform_filter1d
import ocrolib
# from skimage.filters import threshold_sauvola
# from numpy.fft import fft2, fftshift

# from skimage.transform import radon
from ocrolib import lstm, normalize_text
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
# import web
# import threading
import time

       
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
args.model = '/root/ocrapp/models/receipt-model-460-700-00590000.pyrnn.gz'
args.inputdir = '/root/ocrapp/tmp/cleanResult/'
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
#     DSHOW("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binaryary])
    return seeds


@checks(SEGMENTATION)
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background"""
    distances,features = morphology.distance_transform_edt(labels==0,sampling=[3,1], return_distances=1,return_indices=1)
    indexes = features[0]*labels.shape[1]+features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances<maxdist)
    return spread

class PagePredictor:
    def __init__(self):
#         self.lock = threading.Lock()
        self.predictor = Predictor()
    def ocrImage(self, imgpath):
        image = ocrolib.read_image_gray(imgpath)
        tt=time.time()
    #     m = interpolation.zoom(image,args.zoom)
    #     m = filters.percentile_filter(m,args.perc,size=(args.range,2))
    #     m = filters.percentile_filter(m,args.perc,size=(2,args.range))
    #     m = interpolation.zoom(m,1.0/args.zoom)
    # #     if args.debug>0: imshow(m,vmin=0,vmax=1,cmap='gray'); ginput(timeout=-1)
    #     w,h = minimum(array(image.shape),array(m.shape))
    #     flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
    # #     if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1,cmap='gray'); ginput(timeout=-1)
    # 
    #     print 'flatten ', time.time() - tt
    #     tt=time.time()
    #             
    #     temp = interpolation.zoom(flat,0.2)
    # #             if args.debug>0: clf(); imshow(temp,vmin=0,vmax=1,cmap='gray'); ginput(1,-1)
    #     fourier = fftshift(fft2(temp))
    #     sh = log(1 + abs(fourier))
    #     peak = amax(sh)
    # #             if args.debug>0: clf(); imshow(sh,vmin=0,vmax=peak,cmap='gray'); ginput(1,-1)
    # #             continue
    #      
    #     h, w = sh.shape
    #     direction = np.empty(sh.shape)
    #     ncount = 0.0
    #     nsum = 0.0
    #     for i in range(h):
    #         for j in range(w):
    #             ii = i - h/2
    #             jj = j - w/2
    #             if (abs(ii) < 6) & (abs(jj) < 6): continue
    #             if sh[i,j] > 0.4* peak:
    #                 direction = arctan2(-ii,jj)/3.14159*180.0
    #                 if direction < 0:
    #                     direction += 180.0
    #                 if (direction < 100) & (direction > 80):
    #                     ncount += abs(fourier[i,j])
    #                     nsum += direction*abs(fourier[i,j])
    #     if ncount != 0:
    #         newangle = (90.0 - nsum/ncount)*float(h)/w
    #         flat = interpolation.rotate(flat, newangle, cval=1.0)
    # #         if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1,cmap='gray'); ginput(1,-1)
    #     mask = threshold_sauvola(flat, 9, 0.04)
    #     binary = where(flat > mask, 0, 1)     
    # 
    #     print 'deskew and binarize ', time.time() - tt
    #     tt=time.time()         
        
        binary = where(image > 0.5, 0, 1)  
    
        binaryary = morph.r_closing(binary.astype(bool), (args.connect,1))
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
    
    
        location_text = []
        for i,l in enumerate(lines):
            binline = psegutils.extract_masked(1-binary,l,pad=args.pad,expand=args.expand)
            if pre_check_line(binline):
                pred = self.linepredictor.predict(binline)
                result = psegutils.record(bounds = l.bounds, text=pred)
                location_text.append(result)
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
            
        ret = ''
        for i, result in enumerate(location_text): 
            if result is not None:   
                ret += normalize_text(pred) + '\n'
    #             ocrolib.write_text(args.outtext+str(i)+".txt",pred)
        return ret

# print 'global'
# predictors = []
# if len(predictors) == 0:
#     for i in range(16):
#         print 'new predictor'
#         predictors.append(PagePredictor())
#                   
# urls = (
#     '/(.*)', 'miniocropus'
# )
# app = web.application(urls, globals())
# 
#     
# 
# class miniocropus:      
#     def __init__(self):
#         global predictors
#         self.predictors = predictors
#             
#     def GET(self, name):
#         if not name: 
#             return ''
#         else:
#             i = 0
#             while(True):
#                 if not self.predictors[i].lock.locked():
#                     self.predictors[i].lock.acquire()
#                     ret = self.predictors[i].ocrImage(str(args.inputdir + name +'.png'))
#                     self.predictors[i].lock.release()
#                     return ret
#                 else:
#                     i += 1
#                     i %= 16
#                     time.sleep(0.05)
                    
                    
if __name__ == "__main__":
    ret = PagePredictor().ocrImage(str(args.inputdir + sys.argv[1] + '.png'))
    with open(sys.argv[2], 'w') as outputfile:
        outputfile.write(ret)
#     print 'main'
#     global predictors
#     if len(predictors) == 0:
#         for i in range(16):
#             print 'new predictor'
#             predictors.append(PagePredictor())
#     app.run()
#     print 'endmain'
#     s = ocrImage(args.inputdir + '1503019474800.png')
#     print s           
        
        
    