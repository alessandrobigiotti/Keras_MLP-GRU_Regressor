import csv
import numpy as np
import matplotlib
import pickle

from base_dir import *

print("localfile: "+str(local_file_path))
print("parent_dir: "+str(parent_dir))
print("base_dir: "+str(base_dir))
print("raw_dir: "+str(raw_tick_data))
print("raw_1min_data: "+str(raw_1min_data))

input_file_list = os.listdir(raw_tick_data)

print("\n file list: "+str(input_file_list))

for input_csv_file in input_file_list:

	# set the right name for the output file
	output_file = input_csv_file.replace("Ticks", "1Min")
	# change the output file extension
	output_file = output_file.replace("csv", "pkl")
	
	input_file_path = os.path.join(str(raw_tick_data), input_csv_file)
	output_file_path = os.path.join(str(raw_1min_data), output_file)
	
	Time = []
	Date = []
	Open = []
	High = []
	Low = []
	Close = []
	Volume = []

	temp = []
	temp_vol = 0

	i = 0
	time = '00:00:00'

	with open(input_file_path, 'r') as csvfile:
		print("processing file: "+str(input_file_path)+"\n")
		spamreader = csv.reader(csvfile, delimiter = ';')
		for row in spamreader:
					
			if (i > 0):
				timeCurRaw = row[0][11:]
				timeCur = timeCurRaw[:5]
				if (timeCur != time):
					date = row[0][:10]
					time_w = row[0][11:]
					time_w_OK = time_w[:8]
					Time.append(time_w_OK)
					Date.append(date)
					time = timeCur
					ask = float(row[1].replace(',','.'))
					bid = float(row[2].replace(',','.'))
					
					val = float("{0:.5f}".format((float(ask) + float(bid)) / 2))
					Open.append(val)
					t = np.array(temp)
					if(len(t) > 0):
						max = -1.00000
						min = 20000.0
						for elem in t:
							if(float(elem) > max):
								max = float(elem) 
							if(float(elem) < min):
								min = float(elem)
						max = float("{0:.5f}".format(max))
						min = float("{0:.5f}".format(min))
						close = float("{0:.5f}".format(t[(len(t)-1)]))
						Close.append(close)
						High.append(max)
						Low.append(min)
						Volume.append(temp_vol)
						temp = []
						a_vol = float(row[3].replace(',','.'))
						b_vol = float(row[4].replace(',','.'))
						temp_vol = (a_vol + b_vol)/2
						temp.append(val)
						min = 20000.0
						max = -1.0000 
						i = i+1
				else:
					ask = row[1].replace(',','.')
					bid = row[2].replace(',','.')
					a_vol = row[3].replace(',','.')
					b_vol = row[4].replace(',','.')
					val = (float(ask) + float(bid)) / 2
					temp_vol = temp_vol + ((float(a_vol) + float(b_vol)) / 2)
					temp.append(val)
			else:
				i = i+1
	np.array(Open)
	np.array(High)
	np.array(Low)
	np.array(Close)
	np.array(Time)
	np.array(Date)
	np.array(Volume)


	out_file = open(output_file_path, 'wb')
	pickle.dump(Close, out_file)
	print("file created: "+str(output_file_path)+"\n")
	out_file.close()

print('The files are ready!')
#print(Close)

