'''
Created on Sep 4, 2017

@author: loitg
'''

from sklearn import tree
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import graphviz
from matplotlib import pyplot as plt


imgpath = '/home/loitg/workspace/receipttest/img/'

if __name__ == '__main__':
    data = pd.read_csv(imgpath + 'line_decision.csv')
    data_simple = data[(data['target'] < 5) & (data['target'] >= 0)].copy()
  
    data_simple['weight'] = data_simple['target']
    data_simple['target'].replace(1, 0,inplace=True)
    data_simple['target'].replace(2, 0,inplace=True)
    data_simple['target'].replace(3, 1,inplace=True)
    data_simple['target'].replace(4, 1,inplace=True)
    data_simple['weight'].replace(1, 0.3,inplace=True)
    data_simple['weight'].replace(2, 0.3,inplace=True)
    data_simple['weight'].replace(3, 0.1,inplace=True)
    data_simple['weight'].replace(4, 0.18,inplace=True)
    
#     col = data[data['target'] == 0]
#     col.hist()
#     plt.show()


#     data_simple['weight'] = (data_simple['target'] + 1)/3

 
    X_train, X_test, y_train, y_test = train_test_split(data_simple.iloc[:,[0,1,2,3,4,5,6,8]], data_simple.iloc[:,7], test_size=0.4, random_state=1309)
    clf = tree.DecisionTreeClassifier(max_depth = 2, class_weight ={0:3,1:1})
    clf.fit(X_train.iloc[:,:7], y_train, sample_weight=X_train.iloc[:,7].values)
    
    Y_predict = clf.predict(X_test.iloc[:,:7])
    print(classification_report(y_test, Y_predict, target_names=['0','1'], sample_weight=X_test.iloc[:,7].values))
 
    outfile = open(imgpath + 'tree_visualize.dot','w')
    dot_data = tree.export_graphviz(clf, out_file=outfile, feature_names = ['mean','max_fft','mean_fft','dodeu','cycle','ratio','height'], class_names=['line','trash']) 
    graph = graphviz.Source(dot_data) 
    graph
    
    
    
# #     temp = amax(line)-line
# #     temp = temp*1.0/amax(temp) 
#     temp = 1- line  
#     cv2.imshow('temp', temp.astype(float))
# #     labels, n = morph.label(temp)
# #     objects = morph.find_objects(labels)
# #      
# #     height = zeros((temp.shape[1]))
# #     width = zeros((temp.shape[1]))
# #     for obj in objects:
# #         pos = (obj[1].start + obj[1].stop)/2
# #         height[pos] = sl.dim0(obj)
# #         value = sl.dim1(obj)
# #         if value < temp.shape[0]:
# #             width[pos] = value
# #     print len(objects)
# #      
# #     height_plot = zeros(temp.shape)
# #     for i, value in enumerate(height):
# #         pos = int(value)
# #         height_plot[-(pos+3):-pos, i] = 1
# #     cv2.imshow('height', height_plot.astype(float)) 
# #     width_plot = zeros(temp.shape)
# #     for i, value in enumerate(width):
# #         pos = int(value)
# #         width_plot[-(pos+3):-pos, i] = 1
# #     cv2.imshow('width', width_plot.astype(float)) 
#     
#     tt = time()
#     project = mean(temp, axis=0)
#     project = uniform_filter1d(project, line.shape[0]/3)
#     val = mean(project)
#     project = abs(fft(project))
#     project = project[3:len(project)/2]
#     
#     idsort = argsort(project)
#     print 'mean :', val
#     cls_file.write(str(val) + ',')
#     val = amax(project)/line.shape[1]
#     print 'max_fft :', val
#     cls_file.write(str(val) + ',')
#     val = mean(project)
#     print 'mean_fft: ', val
#     cls_file.write(str(val) + ',')
#     val = 1.0*amax(project)/mean(project)
#     print 'dodeu: ', val
#     cls_file.write(str(val) + ',')
#     val = 1.0*line.shape[1]/(idsort[-1] + 3)/line.shape[0]
#     print 'cycle: ', val
#     cls_file.write(str(val) + ',')
#     val = 1.0*line.shape[1] / line.shape[0]
#     print 'ratio: ', val
#     cls_file.write(str(val) + ',')
#     val = line.shape[0]
#     print 'height: ', val
#     cls_file.write(str(val) + ',')
#     
#     print 'time ------------------- : ', time() - tt
# #     project_plot = zeros((100, len(project)))
# #     for i, value in enumerate(project):
# #         pos = int(95*value/amax(project))
# #         project_plot[-(pos+3):-pos, i] = 1
# #     cv2.imshow('fft', project_plot.astype(float)) 
#  
# #     project = mean(temp, axis=0)
# #     project = uniform_filter1d(project, 3)
# #     maxproj = uniform_filter1d(maximum_filter1d(project, line.shape[0]/3),line.shape[0]/2)
# #     minproj = uniform_filter1d(minimum_filter1d(project, line.shape[0]/3),4)
# #   
# #     project_plot = zeros((100, len(maxproj)))
# #     for i, value in enumerate(maxproj):
# #         pos = int(95*value)
# #         project_plot[-(pos+3):-pos, i] = 1
# #     cv2.imshow('max', project_plot.astype(float))  
# #     
# #     project_plot = zeros((100, len(minproj)))
# #     for i, value in enumerate(minproj):
# #         pos = int(95*value)
# #         project_plot[-(pos+3):-pos, i] = 1
# #     cv2.imshow('min', project_plot.astype(float)) 