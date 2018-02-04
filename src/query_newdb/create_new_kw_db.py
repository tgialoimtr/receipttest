'''
Created on Sep 17, 2017

@author: loitg
'''

import re
import pandas as pd


def unicode2ascii(text):
    ret = ''.join(i for i in text if ord(i)<128)
    return ret.encode('utf-8')

def standardize(unitext):
    temp = unicode2ascii(unitext).strip()
    temp = temp.upper()
    temp = re.sub(' +',' ', temp)
    return temp


def gstStandardize(gst):
    temp =  gst.upper()
    temp = re.sub('-','', temp)
    temp = re.sub(' +','', temp)
    if '|' in temp:
        ts = temp.split('|')
        temp = ts[0]
    if (temp == 'same 41') | (temp == 'NO GST'):
        return ''
    return temp
    
if __name__ == '__main__':
#     trung_path = '/home/loitg/Downloads/database_11-09-2017.json'
#     trung = pd.read_json(trung_path)
#     xx=[u'gstNoPattern', u'locationCode', u'mallName', u'receiptIdLastToken', u'receiptIdPattern', u'storeName', u'zipcode']
#     trung['mall0'] = trung['mallName'].apply(standardize)
#     trung['store0'] = trung['storeName'].apply(standardize)
#     trung['mall_kw'] = trung['mall0']
#     trung['store_kw'] = trung['store0']

    trung_path = '/home/loitg/trung_kw_2.csv'
    trung = pd.read_csv(trung_path)   
    
    mall_kw = pd.read_csv('/home/loitg/malls_keyword_edit.csv', header=None, names=['locationCodesList','oriName','kwName'])
    store_kw = pd.read_csv('/home/loitg/stores_keyword_edited.csv', header=None, names=['locationCodesList','oriName','kwName'])
    store_kw.fillna('',inplace=True) 
    mall_kw.fillna('',inplace=True) 
    mall_map = {}
    for i, row in mall_kw.iterrows():
        for lc in row['locationCodesList'].split(' '):
            temp = str(row['kwName']).strip()
            temp = re.sub(' +',' ', temp)
            if len(temp) > 3:
                mall_map[lc] = temp
            else:
                mall_map[lc] = ''
             
    store_map = {}
    for i, row in store_kw.iterrows():
        for lc in row['locationCodesList'].split(' '):
            temp = str(row['kwName']).strip()
            temp = re.sub(' +',' ', temp)
            if len(temp) > 3:
                store_map[lc] = temp
            else:
                store_map[lc] = ''
         
    for i, row in trung.iterrows():
#         row['mall_kw'] = mall_map[row['locationCode']]
        trung.set_value(i,'mall_kw',mall_map[row['locationCode']])
        trung.set_value(i,'store_kw',store_map[row['locationCode']])
         
         
    trung.to_csv('/home/loitg/trung_kw_3.csv', columns=['locationCode','gstNoPattern', 'zipcode','store0','store_kw','mall0','mall_kw','receiptIdLastToken', 'receiptIdPattern'], index=False)

#     trung = pd.read_csv('/home/loitg/trung_kw.csv')
#     trung.fillna('', inplace=True)
#     trung['gstNoPattern'] = trung['gstNoPattern'].apply(gstStandardize)
#     trung.to_csv('/home/loitg/trung_kw_2.csv', index=False)
    