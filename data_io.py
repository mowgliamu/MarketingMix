#!/usr/bin/env python
# coding: utf-8

# # Module Name: data_io
# 
# This module is a part of the **data_preparation** package.
# It contains the neccessary functions required for the following transformations:
# 1. <b>read & write files</b> - common functions which can read & write files, from & to various formats
# 2. <b>stitch data</b> <i>(to be defined)</i> - stitch multiple files to create consolidated file
# 3. <b>set params</b> - read the dictionaries containing parameters and extract the relevant ones 


import pandas as pd
import numpy as np
import os
import math
import openpyxl
from pprint import pprint
from datetime import datetime


# # SECTION: Read & Write Files
# 
# Pandas can 'read' and 'write to' the following (not exhaustive) different types of files:
# 1. Comma-separated values (CSV)
# 2. XLSX
# 3. ZIP
# 4. Plain Text (txt)
# 5. JSON
# 6. XML
# 7. HTML
# 8. Images
# 9. Hierarchical Data Format
# 10. PDF
# 11. DOCX
# 12. MP3
# 13. MP4
# 14. SQL
# ...and more
# 
# These are common functions to 'read' or 'write to' various file format from any given path. 
# 
# <b>NOTE:</b> To be updated as needed


def read_file(path_read, file_format='csv', ret_obj_type='df', **read_params):
    """
    Read files (from any of the given formats) to store in any given object type.
    
    This is a common function to read files from a list of given file formats (default=csv) given a 'path_read' 
    and save as any given object types (default=pandas DataFrame), if supported.
. 
    
    Parameters
    ----------
    path_read : str
        The path where the file to be read is stored.
    file_format : str, default 'csv', other possible 'xlsx'
        The format of the file which is to be read.
    ret_obj_type : str, default 'df', other possible 'dict'
        The object type which will be returned and which stores the data from the read file.
    **read_params
        These parameters will be passed to the corresponding read_file functions as additional information.
    
    Returns
    -------
    obj, default pandas.DataFrame
        Object which stores the data from the read file.
    
    See Also
    --------
    only used when reading 'xlsx' files: sheet_name=0 is the default value.
    
    Examples
    -------
    >>> read_file(path_input); path_input stores the 'full path name' where the file to be read is stored
    df_output, pandas.DataFrame object
    >>> read_file(path_input, 'xlsx', 'dict', sheet_name='spends')
    dict_output, dict object, reading the 'spends' sheet from the given xlsx
    """
    if file_format == 'csv':
        df_output = pd.read_csv(path_read, **read_params)
    elif file_format == 'xlsx':
        df_output = pd.read_excel(path_read, **read_params)
    else:
        print("Error: The given input file format not supported for now.")
    
    if ret_obj_type == 'df':
        return df_output
    elif ret_obj_type == 'dict':
        dict_output = df_output.to_dict()
        return dict_output
    else:
        print("Error: The given output object type not supported for now.")



def write_to_file(df, path_write, file_format='csv', append=False, **write_params):
    """
    Write any dataframe to a file (in any of the given formats).
    
    This is a common function to write any given pandas DataFrame to a file in the given 'path_write',
    in any of given file formats (default=csv), if supported.
. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        The pandas DataFrame which is to be written.
    path_write : str
        The path where the file is to be written and this includes the name of the file, without the extension.
    file_format : str, default 'csv', other possible 'xlsx'
        The format in which the file is to be written.
    append : bool, default False
        Specifies if the files has to be appended to an existing file (e.g. excel worksheet) or not
    **write_params
        These parameters will be passed to the corresponding write_file functions as additional information.
    
    Returns
    -------
        
    See Also
    --------
    prints a 'successfully written' or 'error in writing' message
    
    Examples
    -------
    >>> write_file(df_output, path_write)
    'The file has been successfully writtent to /Desktop/user/output/output.csv'

    >>> write_file(df_output, path_write, file_format='yaml')
    'Error: Sorry, this file format is currently not supported'
    """    
    file_name = path_write + "." + file_format
    
    success_message = "The file has been successfully written to " + str(file_name)
    error_message = "Error: Sorry, this file format is currently not supported"

    if file_format == 'csv':
        df.to_csv(file_name, **write_params)
        print(success_message)
        
    elif file_format == 'xlsx':
        if append is False:
            df.to_excel(file_name, **write_params)
            
        if append is True:
            with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, **write_params)
        print(success_message)
        
    else:
        print(error_message)


#  # SECTION: Stitch Data


def stitch_data(col_anchor, *path_input):
    df_output = pd.DataFrame()
    ...
    return df_output


#  # SECTION: Set Params


def get_params(var, dict_params):
    """
    Get params for a given variable from a master dictionary.
    
    This function finds the given variable in a master dictionary to return a dictionary of corresponding params.
    In case the variable is not found, then it returns a dictionary of default params.
. 
    
    Parameters
    ----------
    var : str
        Name of the variable whose params need to be found.
    dict_params : dict of str : dict
        Master dictionary where params of all variables are stored.
    
    Returns
    -------
    dict of str: float
        Dictionary of the params for the given variable.
        
    Examples
    -------
    >>> get_params('Sales', dict_params_asl_trial)
    dict_params_sales
    """
    if var in dict_params.keys():
        return dict_params[var].copy()
    else:
        return dict_params['default'].copy()




