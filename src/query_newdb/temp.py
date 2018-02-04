'''
Created on Oct 8, 2017

@author: loitg
'''
from fuzzywuzzy import fuzz

top90csv_path = '/home/loitg/Downloads/top90.csv'
desc_path = '/home/loitg/workspace/receipttest/rescources/stores_desc.csv'
mall_map = {
    'Tampines Mall':'TAMPINES MALL|TAMPINES CENTRAL',
    'The Star Vista':'STAR VISTA|1 VISTA EXCHANGE GREEN',
    'IMM Building':'IMM BUILDING|IMM MALL|IMM BRANCH|IMM BLDG',
    'Bukit Panjang Plaza':'BUKIT PANJANG PLAZA|1 JELEBU ROAD',
    'Lot One':'LOT 1|LOT ONE|CHOA CHU KANG',
    'Junction 8':'JUNCTION 8|9 BISHAN PLACE',
    'Bedok Mall':'BEDOK MALL|311 NEW UPPER',
    'Westgate':'3 GATEWAY DRIVE|WESTGATE',
    'Sembawang Shopping Centre':'SEMBAWANG SHOPPING|604 SEMBAWANG',
    'Plaza Singapura':'PLAZA SINGAPURA|68 ORCHARD ROAD',
    'Bugis Junction':'BUGIS JUNCTION|201 VICTORIA|230 VICTORIA',
    'Raffles City Singapore':'RAFFLES CITY|NORTH BRIDGE',
    'JCube':'JCUBE MALL'
    }
if __name__ == '__main__':
    top86_file = open(desc_path,'r')
    top90csv_file = open(top90csv_path,'w')
    top90csv_file.write('Rank,CRM LocationCode,CRM Location Name,Text to identify Mall,Text to identify Merchant,ZipCode to identify Mall,GST to identify Merchant\n');
    
    lines = []
    for line in top86_file:
        lines.append(line.rstrip('\n'))
        
        
        
#     while True:
#         query = raw_input('SEARCH: ')
#         for line in lines:
#             if query.upper() in line.upper():
#                 print line
        
    i = 0
    for line in lines:
        vals = line.split(';')
        store_kws = vals[4].split('|')
        i += 1
        top90csv_file.write(str(i) + ',' + vals[0] + ',' + ',' + mall_map[vals[1]] + ',' + vals[4] + ',' + vals[2] + ',' + vals[3] + '\n')
        
        continue
        for kw in store_kws:
            kw = kw.strip()
            for restline in lines:
                sim = fuzz.partial_ratio(kw.upper(), restline.upper())
                if (sim > 80) and (restline != line):
                    print kw,'---',restline
        k=raw_input()



#     for line in lines:
#         line = line.replace(',',';')
#         vals = line.split(';')
        
        




