# -*- coding: utf-8 -*-
'''
Created on Oct 7, 2017

@author: loitg
'''

import os
import pandas as pd
import shutil
import codecs

part1_path = '/home/loitg/Downloads/part1/'
partall_path = '/home/loitg/Downloads/part2/'
top900path = '/home/loitg/Downloads/top900.xlsx'
desc_path = '/home/loitg/workspace/receipttest/rescources/stores_desc.csv'
arranged_path = '/home/loitg/workspace/receipttest/rescources/arranged/'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
if __name__ == '__main__':
    top900db = pd.read_excel(top900path, sheetname=1)
    with codecs.open(desc_path, 'a+', encoding='utf-8') as outfile:
        sorted_filelist = sorted(os.listdir(part1_path))
        found = False
        for filename in sorted_filelist:
            #next to process 1501683577169
            if filename[:13] != '1501683577169' and (not found):
                continue
            else:
                found = True
            print filename
            rs = None
            while True:
                rs = None
                query = raw_input('input: ')
                if query == 'n':break
                if query == 'q':
                    outfile.close()
                    exit(0)
                names = query.split(';')
                mallname = ''
                locationcode = ''
                storename = names[0]
                if len(names) > 1: mallname = names[1]
                if len(names) > 2: locationcode = names[2]
                
                rs = top900db[top900db['Tenant Name'].str.contains(storename, case=False) &\
                               top900db['Mall SHORT'].str.contains(mallname, case=False) &\
                               top900db['Location Code'].str.contains(locationcode, case=False) ]
                print rs[['Tenant Name', 'Mall SHORT', 'Location Code']]
                if rs is not None and rs.shape[0] == 1:
                    answer = raw_input('COPY TO FOLDER ??? (y/n)')
                    if answer == 'y':
                        break
            
            
            if rs is not None and rs.shape[0] == 1:
                lo = rs['Location Code'].iloc[0]
                ma = rs['Mall SHORT'].iloc[0]
                st = rs['Tenant Name'].iloc[0]
                mallpath = arranged_path + ma + '/'
                locpath = mallpath + lo + '/'
                ensure_dir(mallpath)
                ensure_dir(locpath)
                shutil.copy(part1_path + filename, locpath)
            
                desc = raw_input('description: ')
                outfile.write(lo + ';' + ma + ';' + st + ';' + desc + '\n' )
            
            
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            