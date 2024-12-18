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

# Load all grids as global variables
grids = {
    "r025": pd.read_csv(get_data_file_path('table_r025.csv')),
    "r05": pd.read_csv(get_data_file_path('table_r05.csv')),
    "r075": pd.read_csv(get_data_file_path('table_r075.csv')),
    "r1": pd.read_csv(get_data_file_path('table_r1.csv')),
    "n5": pd.read_csv(get_data_file_path('table_n5.csv')),
    "n10": pd.read_csv(get_data_file_path('table_n10.csv')),
    "n15": pd.read_csv(get_data_file_path('table_n15.csv')),
    "n20": pd.read_csv(get_data_file_path('table_n20.csv')),
}

def calc(x, y, grid_name="n15"):
    """
    Calculate the closest grid point and return related values.
    
    Parameters:
        x (float): [NII]/Ha in log space.
        y (float): [OIII]/Hb in log space.
        grid_name (str): Name of the grid to use (default is 'n15' which grabs the 15 nearest points). Other grids are: 'r025', 'r05', r075', 'r1', 'n5', 'n10', or 'n20'. 
    
    Returns:
        pd.Series: Closest grid point values (f_mean, f_median, f_std, f_var).
    """
    # Get the grid by name
    if grid_name not in grids:
        raise ValueError(f"Grid '{grid_name}' not found. Available grids: {list(grids.keys())}")
    
    grid = grids[grid_name]
    
    # Find the index of the closest x and y pair
    index = ((grid['x'] - x)**2 + (grid['y'] - y)**2).idxmin()
    
    # Retrieve values for the closest pair
    result = grid.loc[index, ['f_mean', 'f_median', 'f_std', 'f_var']]
    return result