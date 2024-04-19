#!/usr/bin/env python
# coding: utf-8

# # Module Name: data_transform
# 
# This module is a part of the **feature_engineering** package.
# It contains the neccessary functions required for the following transformations:
# 1. **change granularity** <i>(to be defined)</i> - transforms the data granularity (e.g. from weekly to monthly level)
# 2. **asl transformation** - transforms any data series (e.g. DataFrame column/s) to a new data series, after applying <i>adstocking</i>, <i>saturation</i> and <i>lag</i>


import os
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from tqdm.auto import tqdm
from pprint import pprint
from joblib import Parallel, delayed

import data_io as io


# Helper Function
def filter_dataframe(datalist, keyword):
    return [val for val in datalist if keyword in val]


# # SECTION: Change Granularity


def change_granularity(df_input, current_granularity, new_granularity):
    """
    This function is yet to be written.
    """
    df_output = pd.DataFrame()
    ...
    return df_output


# # SECTION: ASL Transform

# ### Adstocking:
# 
# Accounts for the **retention-effect** due to the <i>recall value</i> of any ad exposure 
# 
# Adstocking: A(t) = X(t) + retention * A(t-1)
# 
# Where,  
#     A(t) = Ad-stocked value at t  
#     X(t) = New exposure at t (variable column as per file shared)  
#     retention = Adstocking rate
# 
# ### Saturation transformation:
# 
# Accounts for the **diminishing-effect** due to the <i>decreasing marginal utility</i> on adding more adstocks 
# 
# Saturation = S(x) = (Current adstocked value) X Saturation Index  
# Saturation Index = (Current adstocked value ^ shape) / (Current adstocked value ^ shape + steepness ^ shape)
# 
# Where,    
#     Current adstock value = Adstocked value for current timestamp(t)  
#     
# ### Lag transformation:
# 
# Accounts for the **lag-effect** due to the <i>delay</i> between ad-exposure and purchase action
# 
# Shift the 'saturation transformed' column down by the number of rows = lag  
#   

#    ### Trial Values:
#    
#    **retention - trial values (typically):**  
#     Start Value = 0.05  
#     End Value = 0.95  
#     Step Size = 0.05  
#  
#    **Shape (alpha) - trial values (typically):**  
#     Start Value = 2  
#     End Value = 4  
#     Step Size = 0.2  
#     
#    **Steepness (gamma) - trial  values (typically):**  
#     Start Value = min of series value   
#     End Value = 200% X max of series value  
#     Step Size = (End Value - Start Value)/Number of Steps
#     Number of Steps = 19
#     
#    **Lag - trial values (typically):** 0, 1, 2, 3

# ### ASL: Base Transformation Functions

# In[4]:


def ret(half_life):
    """
    Calculate 'retention' from 'half life'.
    
    This function calculates 'retention' whenever the 'half life' of any channel is given. 
    
    Parameters
    ----------
    half_life : float
        The time in which the ad recall reduces to half. Should have same unit as the data granularity.
    
    Returns
    -------
    float
        'Retention', i.e. the % recall of the ads executed in any period, which is retained in the next period
    
    Examples
    -------
    >>> ret(2)
    0.707
    >>> ret(1)
    0.5
    """
    return (0.5)**(1/half_life)



def ads(curr_series_value, prev_adstock, dict_params):
    """
    Calculate 'adstocked value', A(t) of any 'series value', X(t).
    
    This function calculates the standalone 'adstocked value', A(t) for the current period,
    given the 'series value', X(t) for the current period, the 'adstocked value', A(t-1) for the previous period,
    and 'retention' (to be retrieved from dict_params).
    
    Parameters
    ----------
    curr_series_value : float
        Series value (e.g. TV GRP) for the period (e.g. week: n) for which adstocked value is being calculated.
    prev_adstock : float
        Adstocked value for the previous preiod (e.g. week: n-1)
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. retention).
    
    Returns
    -------
    float
        'Adstocked value' for the period
    
    Examples
    -------
    >>> ads(10, 20, dict_params), retention=0.5, as extracted from dict_params.
    20
    """
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    retention = dict_tmp['retention']
        
    result = curr_series_value + (retention * prev_adstock)
    
    return result



def sat(curr_adstock, dict_params):
    """
    Calculate 'saturation transformed value', S(t) of any 'series value', X(t).
    
    This function calculates the standalone 'saturation transformed value', S(x) for any 'adstocked value', x
    given the 'shape (alpha)' and 'steepness (gamma)' (to be retrieved from dict_params). 
    In this case the 'input value' is considered after the 'adstock transformation' of the original series value.
    
    Parameters
    ----------
    curr_adstock : float
        Adstocked value of the series value (e.g. TV GRP) for the same period (e.g. week: n).
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. shape, steepness).
    
    Returns
    -------
    float
        'Saturation-transformed value' for the period
    
    Examples
    -------
    >>> sat(100, dict_params), shape=3, steepness=50, as extracted from dict_params.
    88.89
    """
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    shape = dict_tmp['shape'] 
    steepness = dict_tmp['steepness'] 
    
    numerator = (curr_adstock ** shape)
    denominator = (curr_adstock ** shape + steepness ** shape)
    result = curr_adstock * numerator/ denominator
    
    return result


# ### ASL: Column Transformation Functions


def ads_col(df_input, col_input, dict_params):
    """
    Create an 'adstocked column' for a given column in a pandas DataFrame.
    
    This function uses the given 'retention' value and calculates the corresponding 'adstocked values', 
    for each value in a given series (input column) and returns a new transformed series (adstocked column).
    
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    col_input : str
        Name of the input column which has to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. retention).
    
    Returns
    -------
    pandas.core.series.Series
        'Adstocked column' with corresponding transformed values for a given series
    
    Examples
    -------
    >>> ads_col(df_raw, 'Offline_TV_GRP', dict_params_asl), 
    ...                   retention=0.6, as extracted from dict_params.
    df_output['Offline_TV_GRP_asl_0.6']
    """
    df_output = pd.DataFrame()
    
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    retention = dict_tmp['retention']
    adstock_col = dict_tmp['adstock_col'] 
    lst_new_col_names = dict_tmp['lst_new_col_names_ads']
       
    # if the name of the newly created transformed column is not specified, then use default nomenclature
    if adstock_col is None:
        adstock_col = str(col_input) + '_asl_' + str(round(retention, 2))
            
    # if specified, append the name of the newly created transformed column to a list: lst_new_col_names
    if lst_new_col_names is not None:
        lst_new_col_names.append(adstock_col)
    
    lst_output = [0] # create a list to store transformed values; reduces code complexity
    
    for idx, val in enumerate(df_input[col_input]):
        curr_series_value = val
        prev_adstock = lst_output[idx]
        
        lst_output.append(ads(curr_series_value, prev_adstock, dict_tmp))
    
    df_output[adstock_col] = lst_output[1:] # assign the list to the transformed column
    
    return df_output[adstock_col]



def sat_col(df_input, col_input, dict_params):
    """
    Create a 'saturation-transformed column' for a given column in a pandas DataFrame.
    
    This function uses the given 'shape' and 'steepness' value and calculates the corresponding 
    'saturation-transformed values', for each value in a given series (input column) and 
    returns a new transformed series (saturation-transformed column).
    
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    col_input : str
        Name of the input column which has to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. shape, steepness).
    
    Returns
    -------
    pandas.core.series.Series
        'Saturation-transformed column' with corresponding transformed values for a given series
    
    Examples
    -------
    >>> sat_col(df_raw, 'Offline_TV_GRP_asl_0.6', dict_params_asl), 
    ...                   shape=3, steepness=50, as extracted from dict_params.
    df_output['Offline_TV_GRP_asl_0.6_3_50']
    """
    df_output = pd.DataFrame()
    
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    shape = dict_tmp['shape']
    steepness = dict_tmp['steepness']
    saturation_col = dict_tmp['saturation_col'] 
    lst_new_col_names = dict_tmp['lst_new_col_names_sat']

    
    # if the name of the newly created transformed column is not specified, then use default nomenclature
    if saturation_col is None:
        saturation_col = str(col_input) + '_' + str(round(shape, 2)) + '_' + str(round(steepness, 2))
        
    # if specified, append the name of the newly created transformed column to a list: lst_new_col_names
    if lst_new_col_names is not None:
        lst_new_col_names.append(saturation_col)
    
    df_output[saturation_col] = sat(df_input[col_input], dict_params)

    return df_output[saturation_col]



def lag_col(df_input, col_input, dict_params):
    """
    Create a 'lag-transformed column' for a given column in a pandas DataFrame.
    
    This function uses the given 'lag' values (sequentially) to calculate the corresponding 
    'lag-transformed values', for each value in a given series (input column) and 
    returns a new transformed series (lag-transformed column).  
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    col_input : str
        Name of the input column which has to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. lag).
    
    Returns
    -------
    pandas.core.series.Series
        'lag-transformed column' with corresponding transformed values for a given series
    
    Examples
    -------
    >>> lag_col(df_raw, 'Offline_TV_GRP_asl_0.6_3_50', dict_params_asl), 
    ...                   lag=2, as extracted from dict_params.
    df_output['Offline_TV_GRP_asl_0.6_3_50_2']
    """
    df_output = pd.DataFrame()
    
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    lag = dict_tmp['lag']
    lag_col = dict_tmp['lag_col'] 
    lst_new_col_names = dict_tmp['lst_new_col_names_lag']

    # if the name of the newly created transformed column is not specified, then use default nomenclature
    if lag_col is None:
        lag_col = str(col_input) + '_' + str(round(lag, 0))
        
    # if specified, append the name of the newly created transformed column to a list: lst_new_col_names
    if lst_new_col_names is not None:
        lst_new_col_names.append(lag_col)
    
    df_output[lag_col] = df_input[col_input].shift(periods=lag, fill_value=0) # shift column by lag and fill '0'
    
    return df_output[lag_col]


# ### ASL: Consolidated Transformation Functions


def asl_col(df_input, col_input, dict_params, params=None):
    """
    Run all three transformations sequentially (adstock >> saturation >> lag) for a given column in a DataFrame.
    
    This function uses the given 'retention', shape', 'steepness' and 'lag' values (sequentially) to calculate
    the corresponding 'asl-transformed values', for each value in a given series (input column) and 
    returns a pandas DataFrame including the new transformed series (asl-transformed column).
    
    There is option to keep or drop the 'input' and/or 'intermediate transformed' columns (e.g. ads, sat or lag)
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    col_input : str
        Name of the input column which has to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters to be used for transformation (e.g. retention, shape, steepness, lag).
    
    Returns
    -------
    pandas.DataFrame
        'asl-transformed column' with corresponding transformed values for a given series
    
    Examples
    -------
    >>> asl_col(df_raw, 'Offline_TV_GRP', dict_params_asl), 
    ...                   retention=0.6, shape=3, steepness=50, lag=2, as extracted from dict_params.
    df_output['Offline_TV_GRP_asl_0.6_3_50_2']
    """

    if params is not None:
        dict_params['retention'] = params[0]
        dict_params['shape'] = params[1]
        dict_params['steepness'] = params[2]
        dict_params['lag'] = params[3]
    else:
        pass

    df_output = pd.DataFrame()
    
    dict_tmp = dict_params.copy() # create a copy of dict_params, to protect it during transformation
    
    keep_input_col = dict_tmp['keep_input_col']
    keep_intermediate_col = dict_tmp['keep_intermediate_col']

    df_output[col_input] = df_input[col_input] # add col_input to df_output; drop later if not needed
    
    # run ads, sat and lag column transformations sequentially and concat to df_output
    df_output = pd.concat([df_output, ads_col(df_input, col_input, dict_params)], axis=1)
    col_input_ads = df_output.keys()[-1]

    df_output = pd.concat([df_output, sat_col(df_output, col_input_ads, dict_params)], axis=1)
    col_input_sat = df_output.keys()[-1]

    df_output = pd.concat([df_output, lag_col(df_output, col_input_sat, dict_params)], axis=1)
    col_input_lag = df_output.keys()[-1]
    
    # if not required, drop input and intermediate (i.e. adstocked and saturated) columns
    if keep_input_col is False:
        df_output.drop([col_input], axis=1, inplace=True)        
    if keep_intermediate_col is not True:
        df_output.drop([col_input_ads, col_input_sat], axis=1, inplace=True)
    
    return df_output



def get_trial_lst(current_series, param, dict_params):
    """
    Generate a list of all trial values for a param given its start value, end value and step_size.
    
    This function picks the start value, end value and step_size for any param (e.g. retention, shape, steepness,
    lag) from a given param dictionary and generates a list of all the trial values for that param.
. 
    
    Parameters
    ----------
    param : str
        Name of param whose list of trial values is to be generated.
    dict_params : dict of str : dict
        Master dictionary where params of all variables are stored.
    
    Returns
    -------
    list of float
        List of all the values to be tried of the given param.
        
    Examples
    -------
    >>> get_trial_lst('retention', dict_params_asl_trial)
    [0.05, 0.10, 0.15, 0.20, 0.25, ..., 0.80, 0.85, 0.90, 0.95]
    """
    # define a list for all trial values for 'retention'
    if param == 'steepness':
        start_value = dict_params[param]['start'](current_series)
        #end_value = dict_params[param]['end']
        # PG: Seems like there was a failed attempt to change steepness
        end_value = dict_params[param]['end'](current_series)
        step_size = (end_value - start_value)/dict_params[param]['num_steps']
    
    else:    
        start_value = dict_params[param]['start']
        end_value = dict_params[param]['end']
        step_size = dict_params[param]['step_size']

    lst_trial_values = np.around(np.arange(start_value, end_value, step_size),2).tolist()
    lst_trial_values.append(end_value)
    
    return lst_trial_values


def asl_combinations(df_input, var, dict_params):
    """
    Generate asl-transformed values for a given variable using all possible trial-values of given params.
    
    This function uses all possible trial combinations of the given 'retention', shape', 'steepness' and 'lag' 
    to calculate the corresponding 'asl-transformed' columns for a given variable and generates a DataFrame.
    
    There is option to keep or drop the 'input' and/or 'intermediate transformed' columns (e.g. ads, sat or lag)
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    var : str
        Name of the variable which has to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters (e.g. start_value, end_value, step_size) to generate combinations 
        of the trial values (e.g. retention, shape, steepness, lag) used for asl transformation.
    
    Returns
    -------
    pandas.DataFrame
        'asl-transformed columns' of the variable for each combination of trial values
    
    Examples
    -------
    >>> asl_combination(df_raw, 'Offline_TV_GRP', dict_params_trial), 
    ...                   retention: start_value=0.05, end_value=0.95, step_size=0.05,
    ...                   shape: start_value=2, end_value=4, step_size=0.2, 
    ...                   steepness: ..., 
    ...                   lag: start_value=0, end_value=3, step_size=1,
    ...                   as extracted from dict_params.
    df_Offline_TV_Total_GRPs 
        (col_names: 
        Offline_TV_Total_GRPs_asl_0.05_2.0_0_0, 
        Offline_TV_Total_GRPs_asl_0.05_2.0_0_1,
        ...)
    """

    dict_params_var = io.get_params(var, dict_params) # extract parameters for the specific variable
    
    current_series = df_input[var]

    # generte trial lists for retention, shape, steepness and lag
    lst_retention = get_trial_lst(current_series, 'retention', dict_params_var) 
    lst_shape = get_trial_lst(current_series, 'shape', dict_params_var)
    lst_steepness = get_trial_lst(current_series, 'steepness', dict_params_var)
    lst_lag = get_trial_lst(current_series, 'lag', dict_params_var)
    
    dict_params_trial = dict_params_var['other_params'].copy() # extract other params
    
    # Create all combionations of ASL paramaters
    params_combinations = [list(i) for i in itertools.product(lst_retention, lst_shape, lst_steepness, lst_lag)]
    
    # Parallel processing using joblib
    num_cores = multiprocessing.cpu_count()
    all_transformations = Parallel(n_jobs=num_cores) (delayed(asl_col)\
                                  (df_input, var, dict_params_trial, x)\
                                            for x in params_combinations)

    # Concatenate and add original column
    df_output = pd.concat(all_transformations, axis=1)
    df_output[var] = df_input[var].copy() # copy the variable in output DataFrame to protect the input DataFrame
    
    return df_output



def run_asl_combinations(df_input, lst_var, dict_params, path):
    """
    Generate asl-transformed values for a list of variables using all possible trial-values of given params.
    
    This function uses all possible trial combinations of the given 'retention', shape', 'steepness' and 'lag' 
    to calculate the corresponding 'asl-transformed' columns for all the variables in a given list of variables
    and generates different .csv files for each variable in the given path.
    
    There is option to keep or drop the 'input' and/or 'intermediate transformed' columns (e.g. ads, sat or lag)
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The input DataFrame having the input column which has to be transformed
    lst_var : list of [str]
        Names of all the variables which have to be transformed
    dict_params : dict of {str : float}
        Dictionary of all the parameters (e.g. start_value, end_value, step_size) to generate combinations 
        of the trial values (e.g. retention, shape, steepness, lag) used for asl transformation.
    
    Returns
    -------
    .csv
        separate files of the 'asl-transformed columns' of each variable for each combination of trial values
    
    Examples
    -------
    >>> run_asl_combination(df_raw, ['Offline_TV_GRP', 'Digital_videos', dict_params_trial, path_to_write), 
    ...                   retention: start_value=0.05, end_value=0.95, step_size=0.05,
    ...                   shape: start_value=2, end_value=4, step_size=0.2, 
    ...                   steepness: ..., 
    ...                   lag: start_value=0, end_value=3, step_size=1,
    ...                   as extracted from dict_params.
    Offline_TV_Total_GRPs.csv 
        (col_names: 
        Offline_TV_Total_GRPs_asl_0.05_2.0_0_0, 
        Offline_TV_Total_GRPs_asl_0.05_2.0_0_1,
        ...)
    Digital_videos.csv 
        (col_names: 
        Digital_videos_asl_0.05_2.0_0_0, 
        Digital_videos_asl_0.05_2.0_0_1,
        ...)
    """
    for var in tqdm(lst_var):

        print(lst_var)
        print('Running ASL combinations for: ', var)
        print()

        df_output = asl_combinations(df_input, var, dict_params)
    
        filename = var
        path_to_write = os.path.join(path, filename)
        io.write_to_file(df_output, path_to_write)

