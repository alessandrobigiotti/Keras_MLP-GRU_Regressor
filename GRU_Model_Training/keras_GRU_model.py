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
from keras.models import Model
from keras.layers.core import Activation, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
from gru_iperparameter_settings import *
import sys
sys.path.insert(0, '../')
from Utils.base_dir import *
from Utils.utils import *

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.enable_eager_execution()

def main():
    
	smoothed_data = input("Do you want to use smoothed data? [Y -> yes / N -> no] \n")
	input_file_list = []
	data_dir = ''
	if(smoothed_data == 'Y' or smoothed_data == 'y'):
		input_file_list = os.listdir(raw_1min_data_smoothed)
		data_dir = raw_1min_data_smoothed
	else: 
		input_file_list = os.listdir(raw_1min_data)
		data_dir = raw_1min_data
	
    # fix random seed for reproducibility 
	np.random.seed(7)
    #print(device_lib.list_local_devices())
    
	# Step 1) load the data: import from pickle files
	close_prices = []
	print('loading the data structures...')

	for input_pickle_file in input_file_list:
		input_file_path = os.path.join(data_dir, input_pickle_file)
		file_array = open(input_file_path, 'rb')
		array_dati = pickle.load(file_array)
		close_prices.append(array_dati)
	close_prices = np.array(close_prices)
	
    # Normalize the date
	min_val, max_val, normal_data = Normalize_Data_DataSet(close_prices)
	
	close_list = []
	for i in range(len(normal_data)):
		for j in range(len(normal_data[i])): 
			close_list.append(normal_data[i][j])
    
	normal_list = np.array(close_list)

	# Divide the Data into Train_Samples and Test_Samples
	train_set, valid_set, test_set = Divide_Data(normal_list)

	# Create the training, validation and testing sets
	train_samples = []
	train_labels = []
	test_samples = []
	test_labels = []
	valid_samples = []
	valid_labels = []
	# Create Training samples and labels
	train_samples, train_labels = CreateDataset(train_set, size_campioni)
	valid_samples, valid_labels = CreateDataset(valid_set, size_campioni)
	test_samples, test_labels = CreateDataset(test_set, size_campioni)
	# specify the network optimizer
	optimizer_used = opt.Nadam(lr=learn_rate, beta_1=beta1, beta_2=beta2, epsilon=None, schedule_decay=sched_decay)

	# reshape the training and test samples and labels
	train_samples = np.reshape(train_samples, (train_samples.shape[0], train_samples.shape[1], 1))
	#train_labels_ok = np.reshape(train_labels_ok, (train_labels_ok.shape[0], 1, 1))
	valid_samples = np.reshape(valid_samples, (valid_samples.shape[0], valid_samples.shape[1], 1))
	#valid_labels_ok = np.reshape(valid_labels_ok, (valid_labels_ok.shape[0], 1, 1))
	test_samples = np.reshape(test_samples, (test_samples.shape[0], test_samples.shape[1], 1))
	#test_labels_ok = np.reshape(test_labels_ok, (test_labels_ok.shape[0], 1, 1))

	#print("train labels shapes: "+str(train_labels_ok.shape))

	# Construct the model    
	model = Sequential()
	# Add the first GRU layer
	model.add(GRU(units = output_dim1, input_shape = (size_campioni, 1), recurrent_activation=recurrent_activ_func1, recurrent_dropout=rec_drop_out, return_sequences=True))
	model.add(Activation(activ_function_1))  
	model.add(Dropout(drop_out)) 
	# Add the second GRU layer
	model.add(GRU(units=output_dim2, recurrent_activation=recurrent_activ_func2, recurrent_dropout=rec_drop_out, return_sequences=True))
	model.add(Activation(activ_function_2))
	model.add(Dropout(drop_out))
	# Add the third GRU layer 
	# Add the fourth GRU layer
	model.add(GRU(units=output_dim3, recurrent_activation=recurrent_activ_func3, recurrent_dropout=rec_drop_out, return_sequences=False))
	model.add(Activation(activ_function_3))
	model.add(Dropout(drop_out))
	# Add an output layer with 1 nodes -> it would contain the prediction
	model.add(Dense(1))

	# compile the model using n-adam Optimizer, and minimum square error loss function for regression problem
	model.compile(optimizer = optimizer_used, loss = loss_function, metrics = [metric_function])
	model.summary()
	# plot the model structure 
	#file = str(input('Insert the file to save the network structure: \n'))
	#plot_model(model, to_file = file, show_shapes=True, show_layer_names=True)

	start = tm.time()

	# print the dimensions of the training and testing samples
	#print('Training and Testing Shapes:')
	#print('Training shape: '+str(train_samples.shape)+'Training shape0: '+str(train_samples.shape[0])+'Training shape1: '+str(train_samples.shape[1]))
	call_backs = EarlyStopping(monitor='loss', patience=pati)
	train_log = model.fit(train_samples, train_labels, batch_size=batch_size, epochs=nb_epoch_train, validation_split=0.0, validation_data = (valid_samples, valid_labels), verbose=2, callbacks = [call_backs])
	# Description: train_samples and train_labels are part of the dataset; 
	# batch_size is the dimension of samples per gradient update (simulate the learning rate)
	# epoch are the number of epochs to train the model, each epoch involves the entire train samples and labels; the process is repeated until the epoch value is reached
	# 
	#print(train_log.history)
	end = tm.time() 
	print('execution time training phase: '+str(end-start)+"\n")
	#print(train_log.history.keys())

	#evaluation = model.evaluate(test_samples_ok, test_labels_ok, batch_size = 128, verbose = 1)
	#print(model.metrics_names)
	#print(evaluation)
	# make prediction over the test_set (samples and labels)
	score2 = model.evaluate(valid_samples, valid_labels, batch_size = batch_size, verbose = 1)
	score1 = model.evaluate(test_samples, test_labels, batch_size = batch_size, verbose = 1)

	print('validation set scores: '+str(score2)+'\n')
	print('test set scores: '+str(score1)+'\n')

	prediction = model.predict(test_samples)
	# print('prediction shape: '+str(prediction.shape))

	close_prediction = DeNormalize(prediction, min_val, max_val)
	labels_close = DeNormalize(test_labels, min_val, max_val)
	mse = metr.mean_squared_error(close_prediction, labels_close)
	mae = metr.mean_absolute_error(close_prediction, labels_close)
	print("SCORE MSE: "+str(mse)+"; SCORE MAE: "+str(mae)+"\n")
	print()

	# direct error evaluation: 
	differences = [] 
	for i in range(len(close_prediction)):
		differences.append(labels_close[i] - close_prediction[i])

	new_diff = np.array(differences)

	plt.plot(train_log.history["loss"], color = 'black', linewidth=0.4, label="training (mse)")
	plt.plot(train_log.history["val_loss"], '-.', color = 'grey', linewidth=0.4, label="validation (mse)")
	plt.title('Training e Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.legend()
	plt.show()

	plt.plot(train_log.history['mean_absolute_error'], color = 'black', linewidth=0.4, label='training (mae)')
	plt.plot(train_log.history['val_mean_absolute_error'], '-.', color = 'grey', linewidth=0.4, label = 'validation (mae)')   
	plt.title('Training e Validation Metrics')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Absolute Error')
	plt.legend()
	plt.show()

	plt.plot(test_labels, color = 'black', linewidth=0.5, label = 'labels')
	plt.plot(prediction, color = 'grey', linewidth=0.5, label = 'predictions')
	plt.title('Predictions')
	plt.ylabel('Price values')
	plt.xlabel('Number of samples')
	plt.legend()
	plt.show()
	  
	plt.plot(new_diff, '--', color = 'black', linewidth=0.1, label = 'prediction error')
	plt.yticks(np.arange(-0.01, 0.015, step=0.005))
	plt.legend()
	plt.show()

	save_model = input('Save the Network? ([Y -> YES / N -> NO]): \n')
	if(save_model == 'Y' or save_model == 'y'):
		model.save(os.path.join(str(trained_gru), "GRU_forecast.h5"))
		afile = open(os.path.join(str(trained_gru), "min_val.pkl"), 'wb')
		pickle.dump(min_val, afile)
		afile.close()
		afile = open(os.path.join(str(trained_gru), "max_val.pkl"), 'wb')
		pickle.dump(max_val, afile)
		afile.close()

	print('min, max valued for normalizations: '+str(min_val)+"; "+str(max_val))
	# free the gpu resources
	Kback.clear_session()

if __name__ == "__main__": 
    main()