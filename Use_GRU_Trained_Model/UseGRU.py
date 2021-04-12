import numpy as np
import matplotlib.pyplot as plt 
import pickle
import math 
import os
import tensorflow as tf
import time as tm
import keras as Ker
import keras.backend as Kback
import keras.optimizers as opt
import sklearn.metrics as metr
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Input, GRU, Dense, Embedding
from keras.models import Model, load_model
from keras.layers.core import Activation, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
import csv
import sys
sys.path.insert(0, '../')
from Utils.base_dir import *
from Utils.utils import *
from GRU_Model_Training.gru_iperparameter_settings import size_campioni

def main(): 
    
	#select the model trained
	trained_model = os.path.join(str(trained_gru), "GRU_forecast.h5")
    model = load_model(config)
	# the input dimension for the trained network (have to be the same used during the contruction and the training of the model)
    input_dim = size_campioni
	
	# insert the number of prediction we need (i.e the forecast step). Values bigger than 3 could produce bad results
    step_forecast = int(input('Insert the step forecast horizon (a value bigger than 3 could produce bad results): \n'))

    file = input("Insert the file of minute oscillations (csv format file!): \n")
    i=0
    Price = []
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ';')
        for row in spamreader:
            i=i+1
            if (i > 1):
                Price.append(float(row[4]))
    array_dati = np.array(Price)
    print(array_dati)

	file_min = open(os.path.join(str(trained_gru), "min_val.pkl"), 'rb')
	min_val = pickle.load(file_min)
	file_max = open(os.path.join(str(trained_gru), "max_val.pkl"), 'rb')
    max_val = pickle.load(file_max)
    
    ok = check_MinMax(array_dati, min_val, max_val)
    if(ok == False): 
        print("Max or Min violated, make a new training for the network!")
        exit()
    
    pattern = Normalize_Data_For_RealTimeUse(array_dati, min_val, max_val)

    path = CreatePattern(pattern, input_dim)
    shape_path = []
    shape_path.append(path)
    shape_path = np.array(shape_path)
    shape_path = np.reshape(shape_path, (shape_path.shape[0], shape_path.shape[1], 1))
    pred_made = []

    for i in range(step_forecast): 
        pred = model.predict(shape_path)
        pred_made.append(pred)
        path = np.roll(shape_path[0], -1)
        shape_path[0][len(shape_path[0])-1] = pred
        print('updated patterns: '+str(shape_path))
    
    print(pred_made)
    print('Denormalized prediction: \n')
    predition = DeNormalize_RealTime(pred_made, min_val, max_val)
    print(predition)

if __name__ == "__main__": 
    main()