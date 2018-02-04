'''
Created on Aug 10, 2017

@author: loitg
'''

import numpy as np
import cv2
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn import svm
import xml.dom.minidom
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
from scipy.ndimage import interpolation
import time
from classify.common import sauvola, ASHOW

imgpath = '/home/loitg/workspace/receipttest/img/'


def extractFeature(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (16,16))
    temp = sauvola(img2, w=9, k = 0.3, reverse=True)
#     ASHOW('sample', temp, waitKey=True)
    return np.asarray(temp).flatten()



def resize(img):
    return 


class BoxImgLoader:
        
    def __init__(self, location, markup_file, subfix):
        self.location = location
        self.loadMarkup(markup_file)
        self.subfix = subfix
        
    def loadMarkup(self, markup_file):
        DOMTree = xml.dom.minidom.parse(markup_file)
        collection = DOMTree.documentElement
        images = collection.getElementsByTagName('image')
        self.markup = {}
        for img in images:
            img_path = img.getAttribute('file')
            boxes = img.getElementsByTagName('box')       
            self.markup[img_path] = []
            for box in boxes:
                b={}
                b['top'] = int(box.getAttribute('top'))
                b['left'] = int(box.getAttribute('left'))
                b['width'] = int(box.getAttribute('width'))
                b['height'] = int(box.getAttribute('height'))
                x = box.getElementsByTagName('label')
                b['label'] = x[0].childNodes[0].data if x else u'line'
                self.markup[img_path].append(b)
        
    def boxes(self):
        for img_name in os.listdir(self.location):
            if len(img_name) < len(self.subfix) or img_name[-len(self.subfix):] != self.subfix:
                continue
#             img_name = img_name[:-len(self.subfix)]
            if img_name not in self.markup:
                continue
            boxes = self.markup[img_name]
            img = cv2.imread(self.location + img_name)
            for box in boxes:
                patch = img[box['top']:(box['top']+box['height']) , box['left']:(box['left']+box['width'])]
                temp = {}
                temp['patch'] = patch
                temp['label'] = box['label']
                temp['parent_img'] = img_name
                print temp['label']
                yield temp


if __name__ == '__main__':
#     loader = BoxImgLoader(imgpath, imgpath + 'markup.xml', 'JPG')
#     img_instances = loader.boxes()
#     X = np.empty((0,16*16), dtype=float)
#     Y = []
#     for patch in img_instances:
#         p = patch['patch']
#         if p.shape[0] < 10 or p.shape[1] < 10: continue
#         X = np.vstack([X,extractFeature(p)])
#         label = 0
#         if patch['label'] == 'middleline':
#             label=1
#         if patch['label'] == 'line':
#             label=2
#         Y.append(label)
#                                       
#     np.save('X.npy',X)
#     np.save('Y.npy',Y)
#  
#     X = np.load('X.npy')
#     Y = np.load('Y.npy')
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# #     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,60,20), random_state=190, verbose=True)
# #     clf = svm.SVC(decision_function_shape='ovr')
#     clf = AdaBoostClassifier(n_estimators=100)
#     clf.fit(X_train,Y_train)
#     joblib.dump(clf, 'model4.pkl')
#                     
#     clf = joblib.load('model4.pkl')
#     print clf.score(X_test, Y_test)
#     exit(0)

  
    clf = joblib.load('model4.pkl')
#     a, b = 'e49.JPG', 32
#     a, b = '62e.JPG', 35
#     a, b = '704.JPG', 46
#     a, b = 'ff0.JPG', 68
#     a, b = 'afa.JPG', 48
#     a, b = 'aea.JPG', 50
    a, b = '10f.JPG', 45
     
    img = cv2.imread(imgpath + a) # 62e(35), 704(46), e49(32) ## 10f(45)
#     img = interpolation.rotate(img,45,mode='constant',reshape=0)
    w = img.shape[1]
    h = img.shape[0]
    dw=b
    dh=b
      
    n=0
    total=0
    for x in range(0,w-dw,4):
        for y in range(0,h-dh,4):
            features = extractFeature(img[y:y+dh, x:x+dw])
            begin = time.time()
#             predicted_label = clf.predict_proba(features.reshape(1,16*16))
            predicted_label = clf.predict(features.reshape(1,16*16))
            end = time.time()
            total += (end - begin)
            n += 1
#             predicted_label = predicted_label * 250
#             cv2.circle(img, (x+dw/2,y+dh/2), 2, (predicted_label[0,0],predicted_label[0,1],predicted_label[0,2])) # green
            
            
            if predicted_label == 1:
                cv2.circle(img, (x+dw/2,y+dh/2), 2, (0,255,0)) # green
            elif predicted_label == 2:
                cv2.circle(img, (x+dw/2,y+dh/2), 2, (0,0,255)) # red
            elif predicted_label == 0:
                cv2.circle(img, (x+dw/2,y+dh/2), 2, (255,0,0)) #blue
      
    print total/n,'*',n,'=',total
    cv2.namedWindow('temp',cv2.WINDOW_NORMAL)
    cv2.imshow('temp',img)
    cv2.waitKey(-1)

#     clf = joblib.load('modela1.pkl')
#     a=clf.coefs_[0]
#     b=clf.coefs_[1]
#     print a.shape
#     for i in range(20):
#         temp = a[:,i]
#         print np.min(temp), np.max(temp)
#         temp += 1
#         temp=temp.reshape((16,16))
#         print b[i,:]
#         cv2.namedWindow('temp',cv2.WINDOW_NORMAL)
#         cv2.imshow('temp',temp)
#         cv2.waitKey(-1)

    
    
