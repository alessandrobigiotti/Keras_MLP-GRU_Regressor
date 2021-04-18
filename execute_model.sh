#! /bin/bash
if [ -f $PWD/execute_model.sh ]
	then
		echo Cleaning data and prepare it for the prediction
		cd ./PrepareData
		python create_quotation.py
		cd ..
		echo Select the model for the prediction
		echo type MLP to load multi layer perceptrion model
		echo type GRU to load gated recurrent unit model
		
		read model_type
		
		echo Load the model for the prediction
		
		if [ $model_type == "MLP" ]
		then 
			cd ./Use_MLP_Trained_Model
			echo loading Multi Layer Perceptron model
			python use_MLP.py
			cd ..
		elif [ $model_type == "GRU" ]
		then
			cd ./Use_GRU_Trained_Model
			echo Loading Gated Recurrent Unit model
			python use_GRU.py
			cd ..
		else 
			echo Insert a valid model to make prediction
			cd ..
			
		fi
else
	echo You are not into the main directory.
	echo Please run the script into the repository directory
fi


