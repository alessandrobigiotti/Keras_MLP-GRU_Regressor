import pickle
import pywt
import numpy as np
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '../')
from Utils.utils import *
from Utils.base_dir import *

#font family for the charts
font = {'family':'sans-serif', 'color':'black', 'weight':'normal', 'size' : 12,}

# prova per stampare le wavelets
def PlotChart(array_data):

    x = np.linspace(0.0, 5.0, 100)
    y = np.cos(2*np.pi*x) * np.exp(-x)
    plt.plot(x,y,'k')
    
    plt.title('Demand exponential decay', fontdict = font)
    plt.text(2,0.65,'$\ cos(2 \pi t) \exp(-t)$', fontdict = font)
    plt.xlabel('time (s)', fontdict = font)
    plt.ylabel('voltage (mV)', fontdict = font)

    plt.subplots_adjust(left=0.15)
    plt.show()


def __main():

	input_file_list = os.listdir(raw_1min_data)
	# a paremeter to decide if show or not the plot chart
	show_plot = input("Would you like to see the smoothed chart? type [Y: yes /N: no] \n")
	for input_pickle_file in input_file_list:
	
		# set the right name for the output file
		output_file = input_pickle_file.replace("1Min", "1Min_Smoothed")
		
		input_file_path = os.path.join(raw_1min_data, input_pickle_file)
		output_file_path = os.path.join(raw_1min_data_smoothed, output_file)
		
		# open the pickle file
		file_array = open(input_file_path, 'rb')
		#load the array from the file
		array_dati = pickle.load(file_array)
		#normalize data
		normal_data = Normalize_Data_Data_Raw(array_dati)
		
		#aplly the wavelet decomposition
		#to do this is necessary to specify some parameters used by the decomposition
		print('Wavelet implemented:')
		print(pywt.wavelist(kind='discrete'))
		wavelet_choosen = input('Insert the wavelet choosen: \n')
		print('\nMaximum level decomposition: ')
		max_level = pywt.dwt_max_level(len(normal_data), wavelet_choosen)
		print('max_level = '+str(max_level))
		level_c = input('Insert the decomposition level (no more than max_level): \n')
		wavelet = pywt.Wavelet(wavelet_choosen)
		print("\nProcessing file: "+str(input_file_path)+"\n")
		coeff = pywt.wavedec(normal_data, wavelet_choosen, mode = 'smooth', level = int(level_c))
		coeff = np.array(coeff)
		# threshold the coefficient using their standard deviation: 
		new_coeff = []
		
		for i in range(len(coeff)):
			new_coeff.append(pywt.threshold(coeff[i], 0.001, 'soft'))
		
		np.array(new_coeff)
		
		# print the len coefficient
		#print('Len coeff: '+str(len(coeff)))
		for i in range(0, len(coeff)):
			if (i == 0):
				approx_num = len(coeff) - 1
				# decomment the following lines to show the approximation coefficient
				#print("Approximation Coefficient: A"+str(approx_num))
				#print(coeff[i])
				#print("\n")
			coeff_num = len(coeff) - i - 1
			# decomment the following lines to see the detail coefficient
			#print("Detail coefficient: D"+str(coeff_num))
			#print(coeff[i])
			#print("\n")

		# first reconstruction mode (FULLY Reconstruction)
		reconstruction = pywt.waverec(new_coeff, wavelet)
		
		# to see the reconstruction data decomment the following lines
		#print("Reconstruction Normalized Data")
		#print(reconstruction)
		
		plt.title('Prezzi Grezzi VS Prezzi Smooth')
		plt.plot(reconstruction, linewidth=0.8, color = 'black', label='prezzi smooth')
		plt.plot(normal_data, linewidth=0.8, color = 'darkgrey', label='prezzi grezzi')
		plt.legend()
		if(show_plot == 'y' or show_plot == 'Y'):
			plt.show()
		
		difference = []
		for i in range(0, len(normal_data)-1):
			diff = normal_data[i] - reconstruction[i]
			difference.append(diff)
		np.array(difference)

		plt.plot(difference, linewidth=0.1, color = 'black')
		plt.yticks(np.arange(-0.01, 0.015, step=0.005))
		plt.title('Errore di Ricostruzione')
		if(show_plot == 'y' or show_plot == 'Y'):
			plt.show()
		
		
		plt.plot(normal_data, linewidth=0.1, color = 'black')
		plt.title('Prezzi di Chiusura Normalizzati', fontdict = font)
		plt.xlabel('Time (t)', fontdict = font)
		plt.ylabel('Closed Prices (x)', fontdict = font)
		plt.subplots_adjust(left=0.15)
		if(show_plot == 'y' or show_plot == 'Y'):
			plt.show()
		
		print("\nSaving the smoothed data: "+output_file_path+"\n")
		afile = open(output_file_path, 'wb')
		pickle.dump(reconstruction, afile)
		afile.close()
	
if __name__ == __main(): 
	__main()
