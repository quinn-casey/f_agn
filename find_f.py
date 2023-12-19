#!/usr/bin/env python
# coding: utf-8

# In[8]:


import  pandas as pd
import numpy as np

dfgrid = pd.read_csv('/Users/quinncasey/pyme/f_agn/data/value_table.csv')

def find_f(x, y):
    # Find the index of the closest x and y pair
    index = ((dfgrid['x'] - x)**2 + (dfgrid['y'] - y)**2).idxmin()
    
    # Retrieve values for the closest pair
    result = dfgrid.loc[index, ['f_mean', 'f_median', 'f_std', 'f_var']]
    
    if result.isna().all():
        print('Error: This point falls outside our model (sorry).')
    
    return result


# In[ ]:




