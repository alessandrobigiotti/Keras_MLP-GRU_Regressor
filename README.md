# Keras_MLP-GRU_Regressor

## Introduction
This is a my contribution for the dissertation of my master degree. In this project I present two kind of non linear regressor to time series forecast. The first one 
is implemented by a classic Multi Layer Perceptron Neural Network (MLP). The second one is implemented by a 
Recurrent Neural Network, Gated Recurrent Unit schema (GRU).
At the beginning I focudes only on Forex prices time series, and after I extended my interest in Crypto/Fiat prices time series.

The project is composed by some parts that allows the user to clena and construct a dataset. Starting from the Tick prices movements (i.e. each price movements registered into a minute), 
construct a 1 minute price movements (one can choise if use Ask, Bid, Close, High or Low price). 
