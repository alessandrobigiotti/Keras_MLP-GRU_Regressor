import numpy as np
import matplotlib.pyplot as plt 
import pickle
import math 
import os
import tensorflow as tf
import keras as Ker
import keras.backend as Kback
import keras.optimizers as opt
import time as tm
import sklearn.metrics as metr
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from tensorflow.python.client import device_lib
from mlp_iperparameter_settings import *
import sys
sys.path.insert(0, '../')
from Utils.base_dir import *
from Utils.utils import *

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.enable_eager_execution()

def main():
    
	# for my purpose i decided to use the not smoothed data, to stress the neural network potential
	input_file_list = os.listdir(raw_1min_data)
	# for use smoothed data decomment the following line
	#input_file_list = os.listdir(raw_1min_data_smoothed)
	
    # fix random seed for reproducibility 
	np.random.seed(7)
    #print(device_lib.list_local_devices())
    
	# Step 1) load the data: import from pickle files
	close_prices = []
	print('loading the data structures...')

	for input_pickle_file in input_file_list:
		input_file_path = os.path.join(raw_1min_data, input_pickle_file)
		file_array = open(input_file_path, 'rb')
		array_dati = pickle.load(file_array)
		close_prices.append(array_dati)
	close_prices = np.array(close_prices)

	# Normalize the data
	min_val, max_val, normal_data = Normalize_Data_DataSet(close_prices)

	# Divide the Data into Train_Samples and Test_Samples
	train_set = []
	val_set = []
	test_set = []
	for i in range(len(normal_data)):
		train, test, val = Divide_Data(normal_data[i])
		train_set.append(np.array(train))
		val_set.append(np.array(val))
		test_set.append(np.array(test))
	train_set = np.array(train_set)
	val_set = np.array(val_set)
	test_set = np.array(test_set)

	# Create the training, validation and testing sets
	train_samples = []
	train_labels = []
	test_samples = []
	test_labels = []
	valid_samples = []
	valid_labels = []
	# Create Training samples and labels
	for i in range(len(train_set)):
		train_samp, train_lab = CreateDataset(train_set[i], size_campioni)
		train_samples.append(np.array(train_samp))
		train_labels.append(np.array(train_lab))
	train_samples = np.array(train_samples)
	train_labels = np.array(train_labels)
	# Create Validation samples and labels
	for i in range(len(val_set)):
		val_samp, val_lab = CreateDataset(val_set[i], size_campioni)
		valid_samples.append(np.array(val_samp))
		valid_labels.append(np.array(val_lab))
	valid_samples = np.array(valid_samples)
	valid_labels = np.array(valid_labels)
	# Create Test samples and labels
	for i in range(len(test_set)):
		test_samp, test_lab = CreateDataset(test_set[i], size_campioni)
		test_samples.append(np.array(test_samp))
		test_labels.append(np.array(test_lab))
	test_samples = np.array(test_samples)
	test_labels = np.array(test_labels)

	# Compose the dataset
	train_samples_ok = []
	train_labels_ok = []
	for i in range(len(train_samples)):
		for j in range(len(train_samples[i])):
			train_samples_ok.append(train_samples[i][j])
			train_labels_ok.append(train_labels[i][j])
	train_samples_ok = np.array(train_samples_ok)
	train_labels_ok = np.array(train_labels_ok)

	test_samples_ok = []
	test_labels_ok = []
	for i in range(len(test_samples)):
		for j in range(len(test_samples[i])):
			test_samples_ok.append(test_samples[i][j])
			test_labels_ok.append(test_labels[i][j])
	test_samples_ok = np.array(test_samples_ok)
	test_labels_ok = np.array(test_labels_ok)

	valid_samples_ok = []
	valid_labels_ok = []
	for i in range(len(valid_samples)):
		for j in range(len(valid_samples[i])):
			valid_samples_ok.append(valid_samples[i][j])
			valid_labels_ok.append(valid_labels[i][j])
	valid_samples_ok = np.array(valid_samples_ok)
	valid_labels_ok = np.array(valid_labels_ok)
	# check if it is correct
	#print('\n training set:\n')
	#print(train_samples)
	#print('\n training labels:\n')
	#print(train_labels)

	optimizer_used = opt.Nadam(lr=learn_rate, beta_1=beta1, beta_2=beta2, epsilon=None, schedule_decay=sched_decay)

	# show the shape of the training sample, it is usefull to correctly fit the model
	print('trainsamples shape: '+str(train_samples_ok.shape))

	# Construct the model    
	model = Sequential()
	# Add the input layer with the same shape of the training samples
	model.add(Dense(input_dim=train_samples_ok.shape[1], units=output_dim1))
	model.add(Activation(activ_function_1))
	model.add(Dropout(drop_out))
	# Add an Hidden layer, activation RELU, dropout = drop_out
	model.add(Dense(units=output_dim2))
	model.add(Activation(activ_function_2))
	model.add(Dropout(drop_out))
	# Add an Hiddel layer with , activation RELU, dropout 0.5
	model.add(Dense(units=output_dim3))
	model.add(Activation(activ_function_3))
	model.add(Dropout(drop_out))
	# Add an output layer with 1 nodes -> for this purpos it will contain the prediction
	model.add(Dense(units=1))

	# compile the model using n-adam Optimizer, and minimum square error loss function for regression problem
	model.compile(optimizer = optimizer_used, loss = loss_function, metrics = [metric_function])
	# the following line show the model structure
	model.summary()
	# plot the model structure 
	#file = str(input('Insert the file to save the network structure: \n'))
	#plot_model(model, to_file = file, show_shapes=True, show_layer_names=True)

	# check the time 
	start = tm.time()
	#print('Training and Testing Shapes:')
	#print('Training shape: '+str(train_samples.shape)+'Training shape0: '+str(train_samples.shape[0])+'Training shape1: '+str(train_samples.shape[1]))
	call_backs = EarlyStopping(monitor='loss', patience=pati)
	train_log = model.fit(train_samples_ok, train_labels_ok, batch_size=batch_size, epochs=nb_epoch_train, validation_split=0.0, validation_data = (valid_samples_ok, valid_labels_ok), verbose=2, callbacks = [call_backs])
	# Description: train_samples and train_labels are part of the dataset; 
	# batch_size is the dimension of samples per gradient update (simulate the learning rate)
	# epoch are the number of epochs to train the model, each epoch involves the entire train samples and labels; the process is repeated until the epoch value is reached
	# 
	#print(train_log.history)
	end = tm.time()
	print("execution training phase: "+str(end-start)+"\n")
	#print(train_log.history.keys())
	'''
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	'''
	plt.plot(train_log.history["loss"], color = 'black', linewidth=0.4, label="training (mse)")
	plt.plot(train_log.history["val_loss"], '-.', color = 'grey', linewidth=0.4, label="validation (mse)")
	plt.title('Training e Validation Loss')
	plt.ylabel('Mean Squared Error')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

	plt.plot(train_log.history['mean_absolute_error'], color = 'black', linewidth=0.4, label='training (mae)')
	plt.plot(train_log.history['val_mean_absolute_error'], '-.', color = 'grey', linewidth=0.4, label = 'validation (mae)')   
	plt.title('Training e Validation Metrics')
	plt.ylabel('Mean Absolute Error')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

	#evaluation = model.evaluate(test_samples_ok, test_labels_ok, batch_size = 128, verbose = 1)
	#print(model.metrics_names)
	#print(evaluation)
	# make prediction over the test_set (samples and labels)
	score2 = model.evaluate(valid_samples_ok, valid_labels_ok, batch_size = 128, verbose = 1)
	score1 = model.evaluate(test_samples_ok, test_labels_ok, batch_size = 128, verbose = 1)

	print('validation set scores: ' +str(score2)+'\n')
	print('test set scores: '+str(score1)+'\n')

	prediction = model.predict(test_samples_ok)
	# print('prediction shape: '+str(prediction.shape))

	#denormalize data
	close_prediction = DeNormalize(prediction, min_val, max_val)
	labels_close = DeNormalize(test_labels_ok, min_val, max_val)
	mse = metr.mean_squared_error(close_prediction, labels_close)
	mae = metr.mean_absolute_error(close_prediction, labels_close)
	print("SCORE MSE: "+str(mse))
	print("SCORE MAE: "+str(mae))
	# direct error evaluation: 
	differences = [] 
	for i in range(len(prediction)):
		differences.append(labels_close[i] - close_prediction[i])

	new_diff = np.array(differences)

	plt.plot(labels_close, color = 'black', linewidth=0.5, label = 'labels')
	plt.plot(close_prediction, color = 'grey', linewidth=0.6, label = 'predictions')
	plt.title('Predizioni')
	plt.ylabel('Prezzi di Chiusura')
	plt.xlabel('Numero di Campioni')
	plt.legend()
	plt.show()
	  
	plt.plot(new_diff, '--', color = 'black', linewidth=0.1, label = 'prediction error')
	plt.yticks(np.arange(-0.05, 0.10, step=0.05))
	plt.title('Errore Predizioni')
	plt.legend()
	plt.show()

	save_model = bool(input('Save the Network? (1 -> YES; 0 -> NO): \n'))
	if(save_model):
		model.save('MLP_Conf.h5')

	# free the gpu resources
	Kback.clear_session()
	print('min max for normalization: '+str(min_val)+"; "+str(max_val))
if __name__ == "__main__": 
    main()