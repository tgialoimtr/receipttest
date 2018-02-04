# -*- coding: utf-8 -*-
'''
Created on Sep 16, 2017

@author: loitg
'''
import re
import pandas as pd


info_path = '/home/loitg/Downloads/new_raw.json'



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


def standardize(unitext):
    temp = unicode2ascii(unitext).strip()
    temp = temp.upper()
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

    trung_path = '/home/loitg/Downloads/database_11-09-2017.json'
    trung = pd.read_json(trung_path)
    xx=[u'gstNoPattern', u'locationCode', u'mallName', u'receiptIdLastToken', u'receiptIdPattern', u'storeName', u'zipcode']
    trung['mall0'] = trung['mallName'].apply(standardize)
    trung['store0'] = trung['storeName'].apply(standardize)
    
    storedicttrung = set(trung['storeName'])
    
    
    
    store_dict = {}
    store_set = set(trung['store0'])
    mall_set = set(trung['mall0'])
    zc_set = set(trung['zipcode'])
    gst_set = set(trung['gstNoPattern'])
      
#     outfile = open('/home/loitg/malls_keyword_edit.csv','w')
#     for store in mall_set:
#         hihi = ' '.join(list(trung[trung['mall0'] == store]['locationCode']))
#         outfile.write(hihi)
#         outfile.write(',')
#         outfile.write(store.replace(',','')+',')
#         outfile.write(store.replace(',','')+'\n')
#     outfile.close()
#     exit(0)
    
    while True:
        q = raw_input('query: ')
        print 'ZIPCODES: '
        for storename in zc_set: 
            if q in str(storename):
                print storename
        print 'GST: '
        for storename in gst_set: 
            if q in str(storename):
                print storename
                pass
            
            
#     while True:
#         q = raw_input('query: ')
#         print 'STORES: '
#         for storename in store_set: 
#             if standardize(q) in storename:
#                 print storename
#         print 'MALLS: '
#         for storename in mall_set: 
#             if standardize(q) in storename:
#                 print storename
#                 pass