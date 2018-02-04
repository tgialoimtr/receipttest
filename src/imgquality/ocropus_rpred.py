'''
Created on Jan 11, 2018

@author: loitg
'''
import ocrolib
from ocrolib import lstm, normalize_text
import numpy as np
import cv2

class Predictor:
    def __init__(self):
        self.network = ocrolib.load_object('/home/loitg/workspace/receipttest/model/receipt-model-460-700-00590000.pyrnn.gz',verbose=1)
        for x in self.network.walk(): x.postLoad()
        for x in self.network.walk():
            if isinstance(x,lstm.LSTM):
                x.allocate(5000)
        self.lnorm = getattr(self.network,"lnorm",None)

    def predict(self, line):
        temp = np.amax(line)-line
        temp = temp*1.0/np.amax(temp)
        self.lnorm.measure(temp)
        line = self.lnorm.normalize(line,cval=np.amax(line))
        line = lstm.prepare_line(line,5)
        pred = self.network.predictString(line)
        pred = ocrolib.normalize_text(pred)
        return pred

