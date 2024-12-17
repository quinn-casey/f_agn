import  pandas as pd
import numpy as np
import os

def get_data_file_path(filename):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(filename))

    # Construct the path to the data file relative to the script location
    data_file_path = os.path.join(script_dir, 'f_agn/data', filename)
    #data_file_path = os.path.join(script_dir, 'data', filename)

    return data_file_path

dfgrid = pd.read_csv(get_data_file_path('value_table.csv'))

def kewl(x, y):
    return y > (0.61/(x - 0.47) + 1.19)

def kauf(x, y):
    return y > (0.61/(x - 0.05) + 1.3)

def kewl_line():
    x=np.linspace(-2, 0.4, 100)
    y = (0.61/(x - 0.47) + 1.19)
    return x, y

def calc(x, y):
    # Find the index of the closest x and y pair
    index = ((dfgrid['x'] - x)**2 + (dfgrid['y'] - y)**2).idxmin()
    
    # Retrieve values for the closest pair
    result = dfgrid.loc[index, ['f_mean', 'f_median', 'f_std', 'f_var']]
    
    # If the x, y location is outside the model, then check if the location is above/below the kewley line. If it is above then f_agn = 1 (no std/variance). Otherwise f_agn = 0 
    if result.isna().all():
        # AGN
        if kewl(x, y) or (x > 0.03030):
            result = pd.Series({'f_mean': 1, 'f_median': 1, 'f_std': 0, 'f_var': 0})
        # Composites
        #if (~kewl(x, y) and kauf(x, y)) or (~kewl(x, y) and (x > 0.03030)):
        if (np.logical_not(kewl(x, y)) and kauf(x, y)) or (np.logical_not(kewl(x, y)) and (x > 0.03030)):
            result = pd.Series({'f_mean': 0.5, 'f_median': 0.5, 'f_std': 0, 'f_var': 0})
        if x > 0.47 and np.logical_not(kewl(x, y)):
            result = pd.Series({'f_mean': 0.99, 'f_median': 0.99, 'f_std': 0, 'f_var': 0})
        # SFGs
        if np.logical_not(kauf(x, y)) and (x < 0.0303030):
            result = pd.Series({'f_mean': 0.01, 'f_median': 0.01, 'f_std': 0, 'f_var': 0})
    
    return result