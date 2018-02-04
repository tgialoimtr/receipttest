'''
Created on Nov 1, 2017

@author: loitg
'''
import os
import cv2
from numpy import linspace, shape
import numpy as np
from classify.common import summarize, ASHOW, sauvola, firstAnalyse
from classify.LinesMgr import LinesMgr
from pytesseract import pytesseract
from ocropus_rpred import Predictor
from scipy.misc import imsave
from time import time


allcmnd = '/home/loitg/Downloads/complex-bg/'
# allcmnd = '/home/loitg/workspace/receipttest/img/'

def ocr(img, config=''):
    """Runs Tesseract on a given image. Writes an intermediate tempfile and then runs the tesseract command on the image.

    This is a simplified modification of image_to_string from PyTesseract, which is adapted to SKImage rather than PIL.

    In principle we could have reimplemented it just as well - there are some apparent bugs in PyTesseract (e.g. it
    may lose the NamedTemporaryFile due to its auto-delete behaviour).

    :param mrz_mode: when this is True (default) the tesseract is configured to recognize MRZs rather than arbitrary texts.
    """
    input_file_name = '%s.bmp' % pytesseract.tempnam()
    output_file_name_base = '%s' % pytesseract.tempnam()
    output_file_name = "%s.txt" % output_file_name_base
    try:
        imsave(input_file_name, img)
        status, error_string = pytesseract.run_tesseract(input_file_name,
                                             output_file_name_base,
                                             lang=None,
                                             boxes=False,
                                             config=config)
        if status:
            errors = pytesseract.get_errors(error_string)
            raise pytesseract.TesseractError(status, errors)
        f = open(output_file_name)
        try:
            return f.read().strip()
        finally:
            f.close()
    finally:
        pytesseract.cleanup(input_file_name)
        pytesseract.cleanup(output_file_name)
        
def findTextInImage(img):
    lines = []
    vis = img.copy()
    channels = cv2.text.computeNMChannels(img)
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1,60,0.000015,0.00013,0.5,True,0.1)
        
        erc2 = cv2.text.loadClassifierNM2('/home/loitg/Downloads/opencv_contrib-3.2.0/modules/text/samples/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2,0.5)
        
        regions = cv2.text.detectRegions(channel,er1,er2)
        
        if len(regions) < 2:
            continue
        rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
#         rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)
        
        #Visualization
        for r in range(0, shape(rects)[0]):
            rect = rects[r]
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
            if rect[2] > 15 and rect[3] > 15:
                lines.append(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :])
    
    #Visualization
#     ASHOW("Text detection result", vis, waitKey=True)
#     cv2.waitKey(-1)
    return vis,lines
    
    
if __name__ == '__main__':
#     img = cv2.imread('/home/loitg/Downloads/tmp/23.png', 0)
#     (h,w) = img.shape
#     rotM = cv2.getRotationMatrix2D((w/2,h/2),0,1)
#     img = cv2.warpAffine(img,rotM,(w,h),borderValue=255)
#     img = cv2.resize(img, (0,0), fx = 2.0, fy = 2.0)
#     cv2.imwrite('/home/loitg/Downloads/tmp/'+ '000a' + '.png', img)
#     img = sauvola(img,reverse = True)*255
#     cv2.imwrite('/home/loitg/Downloads/tmp/'+ '000b' + '.png', img)
#     cv2.imshow('gg',img)
#     pred = ocr(img, config='--oem 0 --psm 7 ')
#     print pred
#     pred = ocr(img, config='--oem 1 --psm 7')
#     print pred 
#     pred = ocr(img, config='--oem 0 --psm 7 -c tessedit_char_whitelist=0123456789')
#     print pred
#     cv2.waitKey(-1)
#     exit(0)
    
    linepredictor = Predictor() 
    
    i = 0
    for filename in os.listdir(allcmnd):
        if filename[-3:] != 'JPG' and filename[-3:] != 'jpg': continue
        print filename
        img = cv2.imread(allcmnd + filename)
        vis,lines = findTextInImage(img)
#         cv2.imwrite(allcmnd + 'scenetext_'+filename+'.jpeg', vis)
        
#         for line in lines:
# #             ASHOW('line', line, waitKey=False)
#             i += 1
# #             cv2.imwrite('/home/loitg/Downloads/tmp/'+ str(i) + '.png', line)
#             line = cv2.resize(line, (0,0), fx = 2.0, fy = 2.0)
#             line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
#             line = sauvola(line,scaledown = 0.3, reverse = True)*255
#             tt = time()
#             pred = ocr(line, config='--oem 0 --psm 7')
#             tesstime = time() - tt
#             tt = time()
#             pred2 = linepredictor.predict(255 - line)
#             ocropustime = time() - tt
#             print i, pred, tesstime
#             print i, pred2, ocropustime
# #             cv2.waitKey(-1)

        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w = img_grey.shape
        img_grey = cv2.normalize(img_grey.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         img_grey = sharpen(img_grey)
        img_bin_reversed = sauvola(img_grey, w=128, k=0.2, scaledown=0.4, reverse=True)
#         ASHOW('ori', img_bin_reversed)
        objects, smalldot, scale = firstAnalyse(img_bin_reversed)
#         dotremoved = removedot(img_bin_reversed, smalldot, scale)
        linesMgr = LinesMgr(img_bin_reversed, cv2.cvtColor((img_grey*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        linesMgr.calc(objects, scale)
        for l in linesMgr.lines:
            line = (l.img*255).astype(np.uint8)
            print 'tess-lstm-WOresize ', l.text1
            print 'tess-legacy-WOresize ', l.text2
            pred2 = linepredictor.predict(line)
            print 'ocropus ', pred2
            cv2.imshow('gg', line)
            cv2.waitKey(-1)

        
        