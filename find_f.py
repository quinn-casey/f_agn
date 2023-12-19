import  pandas as pd
import numpy as np
import os

def get_data_file_path(filename):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(filename))

    # Construct the path to the data file relative to the script location
    data_file_path = os.path.join(script_dir, 'data', filename)

    return data_file_path

dfgrid = pd.read_csv(get_data_file_path('value_table.csv'))

def calc(x, y):
    # Find the index of the closest x and y pair
    index = ((dfgrid['x'] - x)**2 + (dfgrid['y'] - y)**2).idxmin()
    
    # Retrieve values for the closest pair
    result = dfgrid.loc[index, ['f_mean', 'f_median', 'f_std', 'f_var']]
    
    if result.isna().all():
        print('Error: This point falls outside our model (sorry).')
    
    return result