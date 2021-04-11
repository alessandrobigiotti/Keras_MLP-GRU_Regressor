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

def NormalizeData(dati, min_val, max_val):
    temp = []
    for i in range (len(dati)):
        temp.append((dati[i] - min_val)/(max_val - min_val))
    
    return(np.array(temp))

def DeNormalize(data, min, max):
    denorm_data = []
    for i in range (len(data)):
        denorm_data.append((data[i][0]*(max - min))+min)
    return(np.array(denorm_data))

def CreatePattern(array, dim):
    dati = []
    for j in range(len(array)-1-dim, len(array)-1, 1):
        dati.append(array[j])

    return(np.array(dati))

def check_MinMax(array_dati, min_val, max_val):
    rel_min = min(array_dati)
    rel_max = max(array_dati)
    
    if(rel_min < min_val or max_val < rel_max):
        return False
    
    return True

def main(): 
    
    config = str(input('Insert the .h5 network model: \n'))
    model = load_model(config)
    input_dim = int(input('Insert the dimension of the input layer: \n'))
    step_forecast = int(input('Insert the step forecast horizon: \n'))

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

    min_val = float(input('Insert the minumum values for the normalization: \n'))
    max_val = float(input('Insert the maximum values for the normalization: \n'))
    
    ok = check_MinMax(array_dati, min_val, max_val)
    if(ok == False): 
        print("Max or Min violated, make a new training for the network!")
        exit()
    
    pattern = NormalizeData(array_dati, min_val, max_val)

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
    predition = DeNormalize(pred_made, min_val, max_val)
    print(predition)

if __name__ == "__main__": 
    main()