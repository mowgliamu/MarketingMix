#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

import data_io as io


def dd_summary(df_input, import_desc='no', import_var_types='no', path_to_write=None):
    """
    Write data dictionary summary as descriptive statistics for all variables.
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The pandas DataFrame input from which summary will be created.
    import_desc: str
        Whether the description of variables is to be imported from a file ("yes" or "no")
    import_var_types: str
        Whether the type of variables is to be imported from a file ("yes" or "no")
    path_to_write : str
        The path where the file is to be written and this includes the name of the file, without the extension.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame which stores the summary for data dictionary.
        
    """    
    dict_dd = {
        'column_name': df_input.columns.tolist(), # list of columns of df_input
    }
    
    df_dd = pd.DataFrame(dict_dd) # empty DataFrame with column-names and their data-types (of df_input) 

    
    # Manual creation of description and category
    if (import_desc or import_var_types) == 'yes' : # if specified, user to upload description/var_type in csv
        if import_desc == 'yes':
            df_dd['description'] = ""
        if import_var_types == 'yes':
            df_dd['type'] = ""
        print('Please enter the column descriptions/variable_type in the mentioned temporary file.')
        io.write_to_file(df_dd, path_to_write, index=False)
        print()

        while True:
            print()
            done = str(input("To upload, please enter 'done' in the box below:")).lower()
            if done == 'done':
                df_dd = io.read_file(path_to_write+'.csv')
                break
            else:
                print()
                print('Try again!! You have NOT entered the correct keyword: "done".') 
                print('Please enter the column descriptions/variable_type in the mentioned temporary file.')
    else:
        pass
    

    # On the fly creation of description and category
    for col in df_dd['column_name']:
        print()
        idx = df_dd[df_dd['column_name']==col].index
        print(idx)
        print(df_input[col].head())
        print(col)

        if import_desc == 'no': # prompt user to write description for each variable
            desc = input('Description: ')
            df_dd.loc[idx, ['description']] = desc
        else:
            pass

        print('Done writing description for', col)

        if import_var_types == 'no': # auto-define var_types
            if str(df_input[col].dtypes) == 'int64' or str(df_input[col].dtypes) == 'float64':
                col_type = 'Numeric'
            else:
                col_type = 'Categorical'            

            df_dd.loc[idx, ['type']] = col_type
        else:
            pass

        print('Done writing variable type for', col)

        # Descriptive Stats
        count_missing = df_input[col].isna().sum()
        count_distinct = df_input[col].nunique(dropna=True)

        if df_dd['type'].values[idx] == 'Numeric':
            min_value = df_input[col].min()
            max_value = df_input[col].max()
            series_range = max_value - min_value
            std_dev = df_input[col].std()

        elif df_dd['type'].values[idx] == 'Categorical':
            min_value = df_input[col].value_counts().idxmin()
            max_value = df_input[col].value_counts().idxmax()
            series_range = "-"
            std_dev = "-"

        else:
            min_value = "-"
            max_value = "-"
            series_range = "-"
            std_dev = "-"

        df_dd.loc[idx, ['count_missing (#)']] = round(count_missing,0)
        df_dd.loc[idx, ['missing_percent (%)']] = round((count_missing/len(df_input[col]))*100,2)
        df_dd.loc[idx, ['count_distinct (#)']] = round(count_distinct,0)
        df_dd.loc[idx, ['distinct_percent (%)']] = round((count_distinct/len(df_input[col]))*100,2)
        df_dd.loc[idx, ['min_val/freq']] = min_value
        df_dd.loc[idx, ['max_val/freq']] = max_value
        df_dd.loc[idx, ['range']] = series_range
        df_dd.loc[idx, ['std_dev']] = std_dev

    df_dd['count_missing (#)'] = df_dd['count_missing (#)'].astype('int')
    df_dd['count_distinct (#)'] = df_dd['count_distinct (#)'].astype('int')
    
    return df_dd


def find_top_cat(df_input, col, num_values=5):
    """
    Get count of unique values using pandas value_counts for a given categorical variable.
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The pandas DataFrame input from which the data will be read.
    col: str
        Column name for which value_counts will be applied.
    num_values : int
        Number of top values which will be returned from value_counts.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output of value_counts.
        
    """    
    df_series = df_input.value_counts(subset=[col], sort=True, ascending=False)[:num_values]

    df_new = pd.DataFrame()
    df_new['variable'] = None
    count = 1

    for name, val in df_series.items():
        var = 'val_' + str(count)
        freq = 'freq_' + str(count)
        df_new[var] = name
        df_new[freq] = val
        count += 1
    df_new['variable'] = col

    return df_new


def dd_categorical(df_input, df_dd_summary, num_values=5):
    """
    Get count of unique values using pandas value_counts for a given categorical variable.
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The pandas DataFrame input from which the data will be read.
    col: str
        Column name for which value_counts will be applied.
    num_values : int
        Number of top values which will be returned from value_counts.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output of value_counts.
        
    """    
    lst_categorical = df_dd_summary['column_name'][df_dd_summary['type'] == 'Categorical'].to_list()

    df_cat = []

    for var in lst_categorical:
        df_cat.append(find_top_cat(df_input, var, num_values))

    df_new = pd.concat(df_cat, axis=0)

    return df_new


def dd_numeric(df_input, df_dd_summary, percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.9999]):
    """
    Get count of unique values using pandas value_counts for a given categorical variable.
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The pandas DataFrame input from which the data will be read.
    col: str
        Column name for which value_counts will be applied.
    num_values : int
        Number of top values which will be returned from value_counts.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output of value_counts.
        
    """    
    lst_numeric = df_dd_summary['column_name'][df_dd_summary['type'] == 'Numeric'].to_list()
    
    df_new = df_input[lst_numeric].describe(percentiles=percentiles).transpose().round(2)
    
    return df_new


def create_data_dictionary(df_input, path_dd, 
                           import_desc='no', import_var_types='no', path_to_write=None,
                           num_values=5,
                           percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.9999]):
    
    """
    Get count of unique values using pandas value_counts for a given categorical variable.
    
    Parameters
    ----------
    df_input : pandas.DataFrame
        The pandas DataFrame input from which the data will be read.
    col: str
        Column name for which value_counts will be applied.
    num_values : int
        Number of top values which will be returned from value_counts.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output of value_counts.
        
    """    
    df_dd_summary = dd_summary(df_input, import_desc, import_var_types, path_to_write)
    
    df_dd_numeric = dd_numeric(df_input, df_dd_summary, percentiles)
    
    df_dd_categorical = dd_categorical(df_input, df_dd_summary, num_values)
    
    print()
    
    io.write_to_file(df_dd_summary, path_dd, file_format='xlsx', sheet_name='Summary')
    io.write_to_file(df_dd_categorical, path_dd, file_format='xlsx', append=True, sheet_name='Categorical')
    io.write_to_file(df_dd_numeric, path_dd, file_format='xlsx', append=True, sheet_name='Numeric')
    
    print()
    
    num_numeric = df_dd_summary['type'].value_counts()['Numeric']
    num_categorical = df_dd_summary['type'].value_counts()['Categorical']
    num_na = df_dd_summary['type'].isna().sum()

    print("Number of Numeric variables:", num_numeric)
    print("Number of Categorical variables:", num_categorical)
    print("Number of Unkown variable types:", num_na)


