import os
import json
import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict
import matplotlib.pyplot as plt
from pprint import pprint

import data_io as io


# Get input data
with open('input_params.json') as f:
    data = json.load(f)

model_config_params = data["model_config_params"]
df_input = pd.read_csv(model_config_params['modeling_data_input'])
target = model_config_params['target_variable']
y_target = df_input[target]
path = model_config_params['output_path']


def get_media_spends_order(spend_keyword='Spend'):
    """
    Get information about media spends for a given dataset.
    
    Parameters
    ----------
    spend_keyword: str
        The keyword which identifies all the media spends variables in the dataset.
    
    Returns
    -------
    Dict
        Ordered dictionary in which media variables are in order of highest spend to lowest spend.
        
    """    

    # Spends DF
    df_spends = df_input.filter(regex=spend_keyword)

    # Sum of spends for all channels, sorted
    budget_allocation = df_spends.sum().sort_values(ascending=False)\
                                 .rename_axis(['Channel']).rename('Spends').reset_index()
    spends_variables = list(budget_allocation['Channel'])   # Ordered list of Spends variables

    # Create dictionary (can be used for forward selection)
    spends_ordered_dict = OrderedDict()
    for var in spends_variables:
        spends_ordered_dict[var] = None

    # Write to file
    total_media_spends = budget_allocation['Spends'].sum()  # Total media spends
    budget_allocation['Spends %'] = (budget_allocation['Spends']/total_media_spends)*100.0      # Percentage of spends for each channel
    path_to_write = os.path.join(path, 'spends_distribution')
    io.write_to_file(budget_allocation, path_to_write)

    return spends_ordered_dict



def media_variables_importance(lst_var):

    """
    Get feature importance through Random Forest / Boruta for a given set of variables.
    
    Parameters
    ----------
    lst_var: str
        List of variables for which feature importances will be evaluated.
    
    Returns
    -------
    Dict, pandas.DataFrame
        Ordered dictionary in order of highest to lowest feature importance.
        DataFrame containing Gini Importance and Boruta Support outout for all the input variables.
        
    """    

    # Get subset of df for which feature importance need to be computed
    df_media = df_input[lst_var]
    y_target = df_input[target]

    # Random Forest
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(df_media, y_target)

    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(df_media.columns, rfr.feature_importances_):
        feats[feature] = importance #add the name/value pair

    rf_importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini_importance_RF'})

    # Create dictionary
    sorted_importance = list(rf_importances.sort_values(by='Gini_importance_RF', ascending=False).index.values)
    rf_ordered_dict = OrderedDict()
    for var in sorted_importance:
        rf_ordered_dict[var] = None
    
    # Boruta object
    boruta = BorutaPy(
    estimator = RandomForestRegressor(n_estimators=100, random_state=42), 
    n_estimators = 'auto',
    max_iter = 100, # number of trials to perform
    perc=50
    )
    
    # Fit
    boruta.fit(np.array(df_media), np.array(y_target))

    # Write Output
    boruta_df = pd.DataFrame(columns=['Feature', 'Boruta_Rank', 'Support', 'Weak_Support'])
    boruta_df['Feature'] = df_media.columns
    boruta_df['Boruta_Rank'] = boruta.ranking_
    boruta_df['Support'] = boruta.support_
    boruta_df['Weak_Support'] = boruta.support_weak_

    boruta_df = boruta_df.set_index('Feature')
    boruta_rf_df = pd.merge(rf_importances, boruta_df, left_on=rf_importances.index, right_on=boruta_df.index)
    boruta_rf_df  = boruta_rf_df.sort_values(['Gini_importance_RF', 'Boruta_Rank'], ascending=[False, False]).reset_index(drop=True)
    boruta_rf_df = boruta_rf_df.rename(columns={'key_0': 'Feature'})

    print(boruta_rf_df)

    # Write to file
    path_to_write = os.path.join(path, 'media_feature_importance')
    io.write_to_file(boruta_rf_df, path_to_write)

    return rf_ordered_dict, boruta_rf_df


def individual_media_asl_importances(lst_var, n_asl, asl_path):

    """
    Get feature importance through Random Forest for all the ASL combinations of a given media variable.
    
    Parameters
    ----------
    lst_var: str
        List of variables for which feature importances will be evaluated.
    n_asl : int
        Number of top most important variables (asl) to be returned.
    asl_path : str
        The path where the ASL dump files will be read from.
    
    Returns
    -------
    Dict
        Ordered dictionary in order of highest to lowest feature importance.
        
    """    

    rf_feat_imp = OrderedDict()
    for var in lst_var:
        current_df = pd.read_csv(os.path.join(asl_path, var + '.csv'))
        # Drop Unnamed and original column
        current_df = current_df.drop(['Unnamed: 0', var], axis=1)
        # Create RF / Fit / Store Importance / Write to file (Top N)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(current_df, y_target)
        current_feat_imp = pd.Series(rf_model.feature_importances_, index=current_df.columns).nlargest(n_asl)
        rf_feat_imp[var] = current_feat_imp

    return rf_feat_imp


def plot_asl_importance(dict_importance, var):

    importances = dict_importance[var]
    asl_var = list(importances.index.values)
    asl_params = []
    for var in asl_var:
        current_asl_params = [float(x) for x in var.split('_')[-4:]]
        asl_params.append(current_asl_params)

    # Create DF
    asl_headers = ['Adstock', 'Alpha', 'Gamma', 'Lag']
    df_asl = pd.DataFrame(asl_params, columns=asl_headers)

    # Plot

    # DEFINE FIGURE SIZE

    plt.subplot(2, 1, 1)
    importances.plot(kind='barh').invert_yaxis()

    plt.subplot(2, 2, 3)
    df_asl['Adstock'].hist(bins=10)

    plt.subplot(2, 2, 4)
    df_asl['Alpha'].hist(bins=10)


    plt.show()

    return

