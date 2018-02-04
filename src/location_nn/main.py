# -*- coding: utf-8 -*-
'''
Created on Sep 10, 2017

@author: loitg
'''

import editdistance
import pandas as pd
from time import time
import string
import os
from math import exp, e
import re

info_path = '/home/loitg/Downloads/new_raw.json'
text_path = '/home/loitg/Downloads/textResult/'
trung_path = '/home/loitg/Downloads/database_11-09-2017.json'

def num_hyphen(names):
#     return names.split('-')[0].strip()'–'
    return names.count('-') + names.count(u'–')

def store(names):
    pos = names.find('-')
    if pos < 0:
        pos = names.find(u'–')
    if pos < 0:
        raise Exception('hihihi')
    return names[:pos].strip()

ignore_line_keyword = ['visa','total', 'date', 'time', 'reg', 'reg.', 'receipt', 'submit', 'charge', 'thank', 'discount', 'description']
similarity_table = string.maketrans('5l1:08O','SIIIDBD')
def standardize(unitext):
    temp = unicode2ascii(unitext).strip()
    temp = temp.translate(similarity_table).upper()
    temp = re.sub(' +',' ', temp)
    return temp

def mall(names):
    pos = names.find('-')
    if pos < 0:
        pos = names.find(u'–')
    if pos < 0:
        raise Exception('hihihi')
    return names[pos+1:].strip()

def containsNumber(name):
    for i in range(10):
        if str(i) in name:
            return True
    return False

class SingleWord:
    def __init__(self, word):
        self.word = word
        self.p = None
        self.count = 1
        
    def setAbsSimilarity(self, val):
        self.p = val

    def updateAbsSimilarity(self, val):
        if val > self.p:
            self.p = val
            
    def getSimilarity(self):
        return 1.0 * self.p / (1+0.05*self.count)
    
def unicode2ascii(text):
    ret = ''.join(i for i in text if ord(i)<128)
    return ret.encode('utf-8')


if __name__ == '__main__':

#     raw = pd.read_json(info_path)
#     stores = raw[(raw.LocationCategory == 'STORE/FOODCOURT') | (raw.LocationCategory == 'STORE/FNB') | (raw.LocationCategory == 'STORE/RETAIL')][['Description','LocationCode','LocationName','Mall']]
#     stores['mall0'] = stores['LocationName'].apply(store)
#     stores['mall0'] = stores['mall0'].apply(standardize)
#     stores['store0'] = stores['LocationName'].apply(mall)
#     stores['store0'] = stores['store0'].apply(standardize)

    
    stores = pd.read_csv('/home/loitg/trung_kw.csv')
    stores.fillna('')
    
#     store_dict = {}
#     store_set = set(stores['store_kw'])
#     for storename in store_set:
#         if len(storename) > 3:
#             for word in storename.split(' '):
#                 if len(word) > 1:
#                     if word in store_dict:
#                         store_dict[word].count += 1
#                     else:
#                         store_dict[word] = SingleWord(word)
# 
#     mall_dict = {}
#     mall_set = set(stores['mall_kw'])
#     for storename in mall_set:   
#         for word in storename.split(' '):
#             if word in mall_dict:
#                 mall_dict[word].count += 1
#             else:
#                 mall_dict[word] = SingleWord(word)

    lines = []
#     pgst1 = re.compile('\W+([MHNl21I][\dOo$DBQRSIl\']-?[\dOoDBQ$SIl\']{7,8}[ ]{0,3}-?[ ]{0,3}\w)\W+')
#     pgst2 = re.compile('([rR][eE][gG]|[gG][$sS5][tT]).*?(\w{1,2}-?\w{6,8}-?\w{1,2})\W')
    pzc1 = re.compile('([sS5][li1I][nN]|[pP][oO0][rR][eE]).*?([\dOoDBQSIl\']{5,7})\W+')
    pzc2 = re.compile('\W(\([S5]\)|[S5][GE]?)[ ]{0,3}([\dOoDBQSIl\']{5,7})\W+')
     
    founds=set()   
    for filename in os.listdir(text_path):
        f = open(text_path+filename,'r')
        print filename, '---------------------------'
        lines = []
        for line in f:
            m1 = pzc1.search(line)
            if m1:
                founds.add(filename)
                print '>>>' , m1.group(2)
                print '----' , line  
            m2 = pzc2.search(line)
            if m2:
                founds.add(filename)
                print '>>>' , m2.group(2)
                print '----' , line        
#         for line in f:
#             m1 = pgst1.search(line)
#             if m1:
#                 founds.add(filename)
#                 print '>>>' , m1.group(1)
#                 print '----' , line
#             m2 = pgst2.search(line)
#             if m2:
#                 founds.add(filename)
#                 print '>>>' , m2.group(2)
#                 print '----' , line
                
            
#     for filename in os.listdir(text_path):
#         if filename not in founds:
#             f = open(text_path+filename,'r')
#             print filename, '---------------------------'
#             for line in f:
#                 print line
    exit(0)
    
            
#         for line in f:
#             cont = False
#             for w in ignore_line_keyword:
#                 if w.upper() in line.upper():
#                     cont = True
#                     break
#             if cont: continue
#             lines.append(line.rstrip('\n'))
#         
#         line_similar = []
#         for i, line in enumerate(lines):
#             for mall in mall_dict.itervalues():
#                 mall.setAbsSimilarity(0.0)
#             for word_line in line.split(' '):
#                 if word_line != '':
#                     similar = 0.0
#                     for word_mall, mall in mall_dict.iteritems():
#                         val = editdistance.eval(word_line.translate(similarity_table).upper(), word_mall)*1.0
#                         x = len(word_mall)*len(word_mall)
#                         val = exp(-val/3)*x/(x+4)
#     #                     val = 1.0 - 1.0*editdistance.eval(standardize(str(word_line)), word_mall)/max([len(word_line), len(word_mall)])
#                         mall.updateAbsSimilarity(val)
#       
#             mallcache = {}
#             for mallname in mall_set:
#                     mallwords = mallname.split(' ')
#                     l = []
#                     for word in mallwords:
#                         l.append(mall_dict[word].getSimilarity())
#                     l.sort(reverse = True)
#     #                 if mallname == 'Sembawang Shopping Centre':
#     #                     print mallname
#     #                     print line
#     #                     print l
#                     val = 0.0
#                     if len(l) > 1:
#                         val = l[0]*l[1]
#                     elif len(l) == 1:
#                         val = l[0]*l[0]
#                     mallcache[mallname] = val
# 
#             result_line = [(value, key) for key, value in mallcache.items()]
#             result_line = max(result_line)
#             neighbours_id = set()
#             for j in range(-4,5,1):
#                 if (i+j >= 0) & (i+j <len(lines)):
#                     neighbours_id.add(i+j)
#             line_similar.append((result_line[0], result_line[1], neighbours_id, line)) 
#          
#          
#         line_similar_store = []
#         line_similar.sort(reverse=True)
#         suspect_stores_range = set()
#         print '-----------'
#         for result in line_similar[:8]:
#             suspect_stores_range.update(result[2])  
#             print result[0], result[1]
#             print result[3]
#   
#         p1 = 0.0
#         p2 = 0.0
#         for line_id in range(len(lines)):
#             for store in store_dict.itervalues():
#                 store.setAbsSimilarity(0.0)
#             tt = time()
#             for word_line in lines[line_id].split(' '):
#                 if word_line != '':
#                     similar = 0.0
#                     for word_store, store in store_dict.iteritems():
#                         val = editdistance.eval(word_line.translate(similarity_table).upper(), word_store)*1.0
#                         x = len(word_store)*len(word_store)
#                         val = exp(-val/3)*x/(x+4)
#     #                     val = 1.0 - 1.0*editdistance.eval(standardize(str(word_line)), word_store)/max([len(word_line), len(word_store)])
#                         store.updateAbsSimilarity(val)
#             p1 += time() - tt
#             tt = time()
#      
#             storecache = {}
#             for storename in store_set:
#                 if len(storename) > 3:
#                     storewords = storename.split(' ')
#                     l = []
#                     for word in storewords:
#                         if len(word) > 1:
#                             l.append(store_dict[word].getSimilarity())
#                     l.sort(reverse = True)
#     #                 if storename == 'Sembawang Shopping Centre':
#     #                     print storename
#     #                     print line
#     #                     print l
#                     val = 0.0
#                     if len(l) > 1:
#                         val = l[0]*l[1]
#                     elif len(l) == 1:
#                         val = l[0]*l[0]/2
#                         
#                     storecache[storename] = val
# 
#             p2 += time() - tt
#             result_line = [(value, key) for key, value in storecache.items()]
#             result_line = max(result_line)
#             line_similar_store.append((result_line[0], result_line[1], lines[line_id])) 
#             
#         line_similar_store.sort(reverse=True)
#         print '-----------' 
#         print p1,p2
#         for result in line_similar_store[:15]:
#             print '++++', result[0], result[1]
#             print '----', result[2]
# 
#   
#         for l in lines:
#             print l 
#   
#     
# 
#             
#     exit(0)
# 
#         
#     print '.'
#         
#     f = open(text_path,'r')
#     for line in f:
#         print line
#     
#     
#     exit(0)
    
# visa
# total
# card
# tax inclusive
# approval code
# date time
# description
# unit
# discount
# charge
# thank
# receipt
# submit
# promo code
# 
# 
# too many symbol (*$+%) over alphanum (abc123)
# too long(llllllllllllllll3l-----------)
# too short(1 A U)
# too many number and '.'
    

            
    pass