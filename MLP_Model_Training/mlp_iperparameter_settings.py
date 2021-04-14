#### created by Alessandro Bigiotti ####
# the iper parameters used by the network, the choise of these values will affect the behaviour of the network
# the dimension of the input layer, is also the number of the prices involved in the prediction
size_campioni = 23
# the dimension of the output for the first layer (the input dimension of the second layer)
output_dim1 = 64
# activation function of the first layer 
activ_function_1 = "tanh"
# the dimension of the output for the second layer (the input dimension of the third layer)
output_dim2 = 32
# activation function of the second layer 
activ_function_2 = "tanh"
# the dimension of the output for the third layer (the input dimension of the fourth layer)
output_dim3 = 16
# activation function of the third layer 
activ_function_3 = "sigmoid"
# the number of epoch on qhich the network will be trained (it will be limited by early_stopping depending on patience configuration)
nb_epoch_train = 100
# the number of samples that will be used for each training step
batch_size = 128
# the dropout value for the hidden payers (express the percentage of weights that will be randomly setted to 0.0)
drop_out = 0.243
# the patience look for loss function change. It will stop the training when the loss change become irrelevant
pati = 6

# optimization parameter for the N-Adam optimizer 
# the learning rate used by the optimization
learn_rate = 0.001
beta1 = 0.9
beta2 = 0.999
sched_decay = 0.004

# loss and metric functions used by the model 
loss_function = "mse"
metric_function = "mae"
