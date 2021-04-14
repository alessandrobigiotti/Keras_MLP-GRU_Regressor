# Keras_MLP-GRU_Regressor

## Introduction
This is my contribution for the master's degree thesis. In this project I present two types of nonlinear regressors for predicting time series. The first one is implemented with a classic Multi Layer Perceptron Neural Network (MLP). The second one is implemented with a Recurrent Neural Network, Gated Recurrent Unit (GRU) scheme.
At first I focused only on Forex price time series and later extended my interest in Crypto / Fiat price time series.

The purpose of this project is to show an easy way to create a good dataset, to train the desired model (MLP or GRU) and to use it in real time forecasting.

## Requirements
If you have a GPU with support to CUDA i strongly recommend to install it. Isn't mandatory, if you haven't such kind of possibility, in the code I've indicated wich line you need to comment to run the program correctly. To install CUDA follow:
* See this [guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4KKVroazE) to install CUDA
* Download and set up [CUDA-Toolkit](https://developer.nvidia.com/cuda-downloads)
To correctly execute the program we need to install and set up:
* _Python3_, _Tensorflow_ and _Keras_. I found a great presentation made by [Jaff Hauton](https://github.com/jeffheaton) using Anaconda, follow the guide [here](https://www.youtube.com/watch?v=dj-Jntz-74g).  If you prefer to use _pip_ then type the following commands (Debian/Ubuntu):
  * $ sudo apt-get update
  * $ sudo apt-get install python3.8
* verify _python_ and _pip_ installation:
  * $ python --version 
  * $ pip --version 
* If the installation is correctly done type:
  * $ pip install --upgrade tensorflow tensorflow-gpu
* _PyWavelets_ is a useful python package needed to apply wavelet transform, [pywt](https://pywavelets.readthedocs.io/en/latest/)

## The program
The project is composed by some python scripts that allow the user to clean and construct a dataset. Starting from the Tick prices movements (i.e. each price movements registered into a minute), construct a 1 minute price movements (one can choose if use Open, Close, High or Low price). Depending on the price choosen, the program will create two time series. The first one will be a sequence of the prices, the second one will be a smoothed version of the same sequence.  After the data is clean, the script start to traing the neural network. One can choose if train the Multi Layer Perceprton or the Gated Recurrent Unit model.  After the network is trained, one can execute the prediction.

### Program description 
* The folder Data contain four folders: 
 * Raw_Tick_Data: containes some _.csv_ files that hold the raw tick prices. The files come from _JForex_ platform that allows to download the tick price movements. [JForex](https://www.dukascopy.com/land/trading/swfx/eu/platforms/?lang=en)
 * Normalized_1Min_Price: containes some pickle files with the normalized price registered into 1 minute
 * Normalized_1Min_SmoothedPrice: containes some pickle files with the normlized prices smoothed by PyWavelets registered into 1 minute
 * Input_Data: contains a _.csv_ file, the input example for the trained network
* The folders MLP_Model_Training and GRU_Model_Training containe both:
 * a file of configuration for the hyper parameters needed by the network (each model has its own configuration)
 * a file that contain the model itself. In there is the DataSet construction and the model training
* The folder PrepareData containe two scripts that, starting from the _.csv_ files contained into Data/Raw_Tick_Data, populate the folders Normalized_1Min_Price and Normalized_1Min_SmoothedPrice.
* The folders Trained_GRU and Trained_MLP containe the trained model (_.h5_ file), and two pickle files min_val.pickle and max_val.pickle that are the max and min values related to the data used for the training. These values will be used to normalize the data for real time usage.
* The folders Use_MLP_Trained_Model and Use_GRU_Trained_Model containe the scripts to read and use the trained model for real time predictions.
* The folder Utils containes some file with utils function and some configurations needed by the project.
* The script _setup.sh_ is the executable, responsible for run all the script needed to clean data, create dataset and train the model. 
* The script _execute_model.sh_ is the executable to load the trained network and make real time prediction.

# Work in progress...

