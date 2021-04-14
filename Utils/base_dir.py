#### created by Alessandro Bigiotti ####
import os
local_file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.join(local_file_path, os.pardir))
base_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
raw_tick_data = os.path.join(str(base_dir), "Data", "Raw_Tick_Data")
raw_1min_data = os.path.join(str(base_dir), "Data", "Raw_1Min_ClosePrice")
raw_1min_data_smoothed = os.path.join(str(base_dir), "Data", "Raw_1Min_ClosePrice_Smoothed")
dataset_dir = os.path.join(str(base_dir), "Data", "Dataset")
trained_mlp = os.path.join(str(base_dir),"Trained_MLP")
trained_gru = os.path.join(str(base_dir),"Trained_GRU")
model_gru = os.path.join(str(base_dir), "GRU_Model_Training")
model_mlp = os.path.join(str(base_dir), "MLP_Model_Training")
