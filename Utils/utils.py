import numpy as np

#max-min to calculate the normalization
min_val = 999999999999
max_val = -999999999999

#function to normalize the prices data
def NormalizeData(resize_data):
    normal_data = []
    global min_val
    min_val = 999999999999
    global max_val
    max_val = -999999999999
    for i in range (0, len(resize_data)-1):
        if resize_data[i] > max_val:
            global max_val
            max_val = resize_data[i]
        if resize_data[i] < min_val:
            global min_val
            min_val = resize_data[i]
    for i in range (0, len(resize_data)-1):
        normal_data.append( (resize_data[i] - min_val)/(max_val - min_val) )
    np.array(normal_data)
    return(normal_data)