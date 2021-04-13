#! /bin/bash

if [ -f $PWD/setup.sh ]
	then
		echo Cleaning data and prepare it for the DataSet construction
		cd ./PrepareData
		python create_raw_close_prices.py 
		python wavelet_prices_smooth.py 
		cd ..
		echo The data are ready
		echo Choose the model to train: 
		echo type MLP to train multi layer perceptrion model 
		echo type GRU to train gated recurrent unit model 

		read model_type 

		if [ $model_type == "MLP" ]
		then 
			cd ./MLP_Model_Training
			echo Training Multi Layer Perceptron model
			python keras_MLP_model.py
		elif [ $model_type == "GRU" ]
		then
			cd ./GRU_Model_Training
			echo Training Gated Recurrent Unit model
			python keras_GRU_model.py
		else 
			echo Insert a valid model to train
			
		fi
else
	echo You are not into the main directory.
	echo Please run the script into the repository directory
fi

