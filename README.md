# Keras_MLP-GRU_Regressor

## Introduction
This is my contribution for the master's degree thesis. In this project I present two types of nonlinear regressors for predicting time series. The first one is implemented with a classic Multi Layer Perceptron Neural Network (MLP). The second one is implemented with a Recurrent Neural Network, Gated Recurrent Unit (GRU) scheme.
At first I focused only on Forex price time series and later extended my interest in Crypto / Fiat price time series.

The purpose of this repository is to show an easy way to create a good dataset, to train the desired model (MLP or GRU) and to use it in real time forecasting.

## Requirements
If you have a GPU with CUDA support I highly recommend you install it. It is not mandatory. If you do not have this kind of possibility, in the code I have indicated which lines you need to comment to correctly run the program. To set up all necessary softwares follow: 
* See this [guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4KKVroazE) to install CUDA
* Download and set up [CUDA-Toolkit](https://developer.nvidia.com/cuda-downloads)
To correctly execute the program we need to install and set up:
* _Python3_, _Tensorflow_ and _Keras_. I found a great presentation made by [Jaff Hauton](https://github.com/jeffheaton) using Anaconda, see the [guide](https://www.youtube.com/watch?v=dj-Jntz-74g).  If you don't want install Anaconda, type the following commands (Debian/Ubuntu):
  * $ sudo apt-get update
  * $ sudo apt-get install python3.8
* verify _python_ and _pip_ installation:
  * $ python --version 
  * $ pip --version 
* If the installation is correctly done type:
  * $ pip install --upgrade tensorflow
* If you have installed CUDA, you can use _tensorflow-gpu_:
  * $ pip install --upgrade tensorflow-gpu
* _PyWavelets_ is a useful python package needed to apply wavelet transform, see [pywt](https://pywavelets.readthedocs.io/en/latest/)  
**NOTICE**: pay attention that your GPU drivers, CUDA-Toolkit and tensorflow-gpu have a compatible version. In general you need a python version greater than 3.4.X to execute the program.

## The program
The project is composed by some python scripts that allow the user to clean data,  to construct a dataset, to train a neural network model and to use it. Starting from the Tick prices movements (i.e. any price movement that has occurred), it constructs a 1 minute price movements (one can choose if use Open, Close, High or Low price). Depending on the price choosen, the program will create two time series. The first one will be a sequence of the prices, the second one will be a smoothed version of the same sequence. After the data is ready, the script start to train the neural network. One can choose if train the Multi Layer Perceprton or the Gated Recurrent Unit model. After the network is trained, one can execute the prediction.

### Program description 
* The _Data_ folder contains four folders: 
  * _Raw_Tick_Data_: containes some _.csv_ files that hold the raw tick prices. The files come from [Dukascopy](https://www.dukascopy.com/land/trading/swfx/eu/home/?lang=en)
    broker, and are downloaded from [JForex](https://www.dukascopy.com/land/trading/swfx/eu/platforms/?lang=en) platform. In the folder you can find the quotations of Bitcoin  
    against US dollar (BTC/USD) related to the January and February 2018.
  * _Normalized_1Min_Price_: contains some pickle files with the normalized price registered into 1 minute. 
  * _Normalized_1Min_SmoothedPrice_: contains some pickle files with the normlized prices smoothed by PyWavelets registered into 1 minute.
  * _Input_Data_: contains a _.csv_ file, the input example for the trained network. It is needed to real time usage.
* The _MLP_Model_Training_ and _GRU_Model_Training_ folders contain both:
  * a file of configuration for the hyper parameters needed by the network (each model has its own configuration)
  * a file that contain the model itself. In the files there is the DataSet construction and the model training.
* The _PrepareData_ folder contains two scripts which, starting from the _.csv_ files contained in Data/Raw_Tick_Data, populate the folders Normalized_1Min_Price and
  Normalized_1Min_SmoothedPrice.
* The _Trained_GRU_ and _Trained_MLP_ folders contain the trained models (the _.h5_ file), and two pickle files min_val.pickle and max_val.pickle that are the max and min values
  related to the data used for the training. These values will be used to normalize the data for real time usage.
* The _Use_MLP_Trained_Model_ and _Use_GRU_Trained_Model_ folders contain scripts to load and use the trained model for real-time forecast.
* The _Utils_ folder containes some file with utils function and some configurations needed by the project.
* The script _setup.sh_ is the executable, responsible for run all the script needed to clean data, create dataset and train the model.
* The script _execute_model.sh_ is the executable to load the trained network and make real time prediction.

## How it functions
Download the repository and move the main directory _Keras_MLP-GRU_Regressor_. Open here the terminal, then run:
* $ ./setup.sh
* $ ./execute_model.sh

The first script will create all the the data needed (normalized data, trained model), then it will perform the training and the validation over the desired neural network. The second script will execute the model trained to perform a real time forecast.  
**NOTICE**: The purpose of this project is for demonstration only. Before using it in a real context it is necessary to structure the real-time forecast data in another way (retrieve data from api, query a DB and so on ...), but it is beyond my intentions here.

