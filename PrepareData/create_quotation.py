### created by Alessandro Bigiotti ###
import csv
import numpy as np
import matplotlib
from builtins import print
import re as r
import sys 
sys.path.insert(0, '../')
from Utils.base_dir import *

ask_pattern = r.compile(r'(Ask){1}')
bid_pattern = r.compile(r'(Bid){1}')

input_file_list = os.listdir(input_data)

file_ask_name = [s for s in input_file_list if ask_pattern.search(s)][0]
file_bid_name = [s for s in input_file_list if bid_pattern.search(s)][0]

file_ask = os.path.join(str(input_data), file_ask_name)
file_bid = os.path.join(str(input_data), file_bid_name)

output_name = file_bid.replace('_Bid_','_Average_')
file_out = os.path.join(str(input_data), output_name)

print(file_ask)
print(file_bid)
print(file_out)
row_bid = []
row_ask = []

Open = []
Close = []
High = [] 
Low = [] 
Volume = []

i = 0

with open(file_bid, 'rt') as csvbid:
	spamreader_bid = csv.reader(csvbid, delimiter=';', quotechar=' ')
	for row in spamreader_bid:
		if ( i > 0):
			row_bid.append(row)
			i = i+1
		else:
			print('Reading file bid; columns names:' + str(row))
			i = i+1
np.array(row_bid)

with open(file_ask, 'rt') as csvask:
	spamreader_ask = csv.reader(csvask, delimiter=';', quotechar=' ')
	i = 0
	for row in spamreader_ask:
		if ( i > 0 ):
			row_ask.append(row) 
			i = i+1
		else: 
			print ('Reading file ask; rows structures:' + str(row))
			i = i+1
np.array(row_ask)

print('Calculating the quotations...')
for i in range(0,len(row_ask)):
		
		val_op = (float(row_ask[i][1].replace(',','.')) + float(row_bid[i][1].replace(',','.'))) / 2.0
		Open.append(val_op)
		
		val_cl = (float(row_bid[i][4].replace(',','.')) + float(row_ask[i][4].replace(',','.'))) / 2.0
		Close.append(val_cl)
		
		val_hi = (float(row_ask[i][2].replace(',','.')) + float(row_bid[i][2].replace(',','.'))) / 2.0
		High.append(val_hi)
		
		val_lo = (float(row_ask[i][3].replace(',','.')) + float(row_bid[i][3].replace(',','.'))) / 2.0
		Low.append(val_lo)
		
		len1 = len(row_ask[i][5])
		len2 = len(row_bid[i][5])
		val1 = str(row_ask[i][5][:(len1-1)])
		val2 = str(row_bid[i][5][:(len2-1)])
		val1 = str(val1.replace(',','.') if val1 != '' else  0)
		val2 = str(val2.replace(',','.') if val2 != '' else  0)
		val = (float(val1) + float(val2)) / 2.0 
		Volume.append(val)

np.array(Open)
np.array(Close)
np.array(High)
np.array(Low)
np.array(Volume)

with open(file_out, 'w') as csvout:
	spamwriter = csv.writer(csvout, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i in range(0, (len(Open))):
		spamwriter.writerow([Volume[i], High[i], Low[i], Open[i], Close[i]])
ind = len(Open)
print(Open[ind-1],High[ind-1],Low[ind-1],Close[ind-1], Volume[ind-1], ind)

date_time = row_ask[i][0][1:]
print(date_time)	

print("Quotation file created! \n")