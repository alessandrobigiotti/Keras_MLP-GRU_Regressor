#### created by Alessandro Bigiotti ####
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

MIN_VAL = 1.11775
MAX_VAL = 1.14858

def NormalizeData(dati):
    temp = []
    for i in range (len(dati)):
        temp.append((dati[i] - MIN_VAL)/(MAX_VAL - MIN_VAL))
    
    return(np.array(temp))

def DeNormalize(data):
    denorm_data = []
    for i in range (len(data)):
        denorm_data.append((data[i][0]*(MAX_VAL - MIN_VAL)) + MIN_VAL)
    return(np.array(denorm_data))

def CreatePattern(array, dim):
    dati = []
    for j in range(len(array)-1-dim, len(array)-1, 1):
        dati.append(array[j])

    return(np.array(dati))

def check_MinMax(array_dati):
    rel_min = min(array_dati)
    rel_max = max(array_dati)
    
    if(rel_min < MIN_VAL or MAX_VAL < rel_max):
        return False
    
    return True

def main(): 
    
    config = str(input('Insert the .h5 network model: \n'))
    model = load_model(config)
    input_dim = int(input('Insert the dimension of the input layer: \n'))
    step_forecast = int(input('Insert the step forecast horizon: \n'))

    while(True):
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
    
        ok = check_MinMax(array_dati)
        if(ok == False): 
            print("Max or Min violated, make a new training for the network!")
            exit()
    
        pattern = NormalizeData(array_dati)

        path = CreatePattern(pattern, input_dim)
        shape_path = []
        shape_path.append(path)

        pred_made = []

        for i in range(step_forecast): 
            pred = model.predict(np.array(shape_path))
            pred_made.append(pred)
            shape_path = np.roll(shape_path, -1)
            #print("rolled path: \n")
            #print(shape_path)
            shape_path[0][len(shape_path[0])-1] = pred
            print('updated patterns: \n')
            print(str(shape_path))
    
        print(pred_made)
        print('Denormalized prediction: \n')
        prediction = DeNormalize(pred_made)
        print(prediction)

if __name__ == "__main__": 
    main()
