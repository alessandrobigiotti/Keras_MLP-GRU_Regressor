import os
local_file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.join(local_file_path, os.pardir))
base_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
raw_tick_data = os.path.join(str(base_dir), "Data", "Raw_Tick_Data")
raw_1min_data = os.path.join(str(base_dir), "Data", "Raw_1Min_ClosePrice")