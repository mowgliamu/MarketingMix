import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import chain
from pprint import pprint

from asl import *


# Get input data
with open('input_params.json') as f:
    data = json.load(f)

model_config_params = data["model_config_params"]
df_input = pd.read_csv(model_config_params['modeling_data_input'])
y_target = df_input[model_config_params['target_variable']]

#===========
# Functions
# ==========

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_vif(x):

    # Get a VIF factor score
    if x.shape[1] == 1:
        vif = [1.0]
    else:
        vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    
    return vif


def get_mape(y_true, y_pred):
    y_true = np.asanyarray(y_true)  # actual
    y_pred = np.asanyarray(y_pred)  # predicted
    mape_val = np.nanmean(np.abs((y_true - y_pred)/y_true))*100

    return mape_val


def regression_results(y_true, y_pred):

    # Evaluation Metrics 
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mae=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    rmse=np.sqrt(mse)
    r2=metrics.r2_score(y_true, y_pred)
    mape=get_mape(y_true, y_pred)

    return explained_variance, r2, mae, mse, rmse, mape


def log_likelihood(y, y_hat):

    resid = np.subtract(y, y_hat)
    sse = np.sum(np.power(resid, 2))
    n = len(y)
    sigma_sq = sse / n
    ll = -n/2 * np.log(2*np.pi*sigma_sq) - 1/(2*sigma_sq) * sse

    return ll


def get_aic(y, y_hat, k):

    ll = log_likelihood(y, y_hat)

    return 2*k - 2*ll


def get_contribution(period, df_decomps):

    dv_sum_period = y_target[-period:].sum()
    dv_sum_rest = y_target[:-period].sum()

    contri_mentioned_period = df_decomps[-period:].sum()
    contri_rest_period = df_decomps[:-period].sum()

    contri_percent_mentioned_period = (contri_mentioned_period/dv_sum_period)*100.
    contri_percent_rest_period = (contri_rest_period/dv_sum_rest)*100.
    
    return contri_mentioned_period, contri_rest_period, contri_percent_mentioned_period, contri_percent_rest_period


def get_contribution_single_model(X_input, beta, period):

    # for period: sum(beta_i*x_i)/sum(target) * 100
    # For baseline: y_target - beta_i*x_i, then do sum

    nvar = X_input.shape[1]

    contribution_period = []
    contribution_rest = []
    contribution_period_pct = []
    contribution_rest_pct = []

    baseline_decomposition = y_target.copy()
    for i in range(1, nvar):
        current_series = X_input.iloc[:, i]
        current_coeff  = beta[i]
        # beta_i * X_i
        decomposed_series = current_series * current_coeff
        # Get contribution
        contri, contri_rest, contri_p, contri_p_rest = get_contribution(period, decomposed_series)
        contribution_period.append(contri)
        contribution_rest.append(contri_rest)
        contribution_period_pct.append(contri_p)
        contribution_rest_pct.append(contri_p_rest)
        # Basline: Y_target - beta*X (So basline contribution is time-dependent now!)
        baseline_decomposition = baseline_decomposition - decomposed_series

    # Baseline contribution
    contri, contri_rest, contri_p, contri_p_rest = get_contribution(period, baseline_decomposition)
    contribution_period.insert(0, contri)
    contribution_rest.insert(0, contri_rest)
    contribution_period_pct.insert(0, contri_p)
    contribution_rest_pct.insert(0, contri_p_rest)

    return contribution_period, contribution_rest, contribution_period_pct, contribution_rest_pct
    

def get_error_metrics(X, y, ols_results):

    # Predict on test set and evaluate metrics
    coeff = ols_results.params
    y_hat = ols_results.predict(X)
    exp_var, r2, mae, mse, rmse, mape = regression_results(y, y_hat)

    vif_info = get_vif(X)
    p_value_limit = model_config_params['p_value_limit'] 
    r_square_limit = model_config_params['r_square_limit'] 
    vif_limit = model_config_params['vif_limit'] 

    # Get Contributions in a specified period
    contri_period = model_config_params['contribution_period']
    cp, cr, cpp, crp = get_contribution_single_model(X.copy(), coeff, contri_period)

    # Flags
    list_model_results = []
    for i in range(X.shape[1]):
        flag_est = [0 if ols_results.params[i] > 0 else 1 ]
        flag_p_value = [0 if ols_results.pvalues[i] < p_value_limit else 1 ]
        list_model_results.append({
            'Dependent': y_target.sum(),
            'Variable': X.columns[i],
            'p-value': ols_results.pvalues[i],
            'R-square': ols_results.rsquared,
            'Adj-R2': ols_results.rsquared_adj,
            'AIC': ols_results.aic,
            'MAPE': mape,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'VIF':vif_info[i],
            'Contribution_percent(last_' + str(contri_period) + ')': cpp[i],
            'Contribution(last_' + str(contri_period) + ')': cp[i],
            'Contribution_percent(rest)': crp[i],
            'Contribution(rest)': cr[i],
            'Estimate':ols_results.params[i],
            't-value':ols_results.tvalues[i],
            'Std_Error':ols_results.bse[i],
            'F-stat':ols_results.fvalue,
            'p-val(F-stat)': ols_results.f_pvalue,
            'DW-stat':durbin_watson(ols_results.resid),
        })

    return y_hat, list_model_results


def create_validation_set(X, y):

    '''
    Backward, first 6 months should be used for validation
    Model will be built on rest
    '''

    modeling_period = model_config_params['modeling_period']
    validation_period = model_config_params['validation_period']
     
    X_train = X[-modeling_period:]
    X_validate = X[:validation_period]

    y_train = y[-modeling_period:]
    y_validate = y[:validation_period]

    return X_train, X_validate, y_train, y_validate


def build_stepwise_regression(X, y):

    # Add constant (for sm)
    X = sm.add_constant(X)

    # Build Model on training set
    model = sm.OLS(y, X)
    ols_results = model.fit()

    # ===============
    # Model Selection
    # ===============

    vif_info = get_vif(X)
    coeff = ols_results.params
    p_value_limit = model_config_params['p_value_limit'] 
    r_square_limit = model_config_params['r_square_limit'] 
    vif_limit = model_config_params['vif_limit'] 
    
    # Separate out competitor and media variables
    var_media, var_comp = [], []
    for var, val_that_caused_me_grief in dict(ols_results.params).items():
        if var.startswith('Competition'):
            var_comp.append(var)
        else:
            var_media.append(var)


    accept_model = False
    if var_comp:
        if all(coeff[var_comp] < 0) and all(coeff[var_media] > 0) and max(ols_results.pvalues) <= p_value_limit and all(vif_info) < vif_limit and round(ols_results.rsquared,3) > r_square_limit:
            accept_model = True
        else:
            pass
    else:
        if all(coeff[var_media] > 0) and max(ols_results.pvalues) <= p_value_limit and all(vif_info) < vif_limit and round(ols_results.rsquared,3) > r_square_limit:
            accept_model = True
        else:
            pass


    # Only proceed further if the model is selected

    if accept_model:

        # Predict on train and validation set and evaluate metrics
        y_hat = ols_results.predict(X)
        exp_var, r2, mae, mse, rmse, mape = regression_results(y, y_hat)

        # Get Contributions in a specified period
        contri_period = model_config_params['contribution_period']
        cp, cr, cpp, crp = get_contribution_single_model(X.copy(), coeff, contri_period)

        # Flags
        list_model_results = []
        for i in range(X.shape[1]):
            flag_est = [0 if ols_results.params[i] > 0 else 1 ]
            flag_p_value = [0 if ols_results.pvalues[i] < p_value_limit else 1 ]
            list_model_results.append({
                #'Model': model_index,
                'Dependent': y_target.sum(),
                'Variable': X.columns[i],
                'p-value': ols_results.pvalues[i],
                'R-square': ols_results.rsquared,
                'Adj-R2': ols_results.rsquared_adj,
                'AIC': ols_results.aic,
                'MAPE': mape,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'VIF':vif_info[i],
                'Contribution_percent(last_' + str(contri_period) + ')': cpp[i],
                'Contribution(last_' + str(contri_period) + ')': cp[i],
                'Contribution_percent(rest)': crp[i],
                'Contribution(rest)': cr[i],
                'Estimate':ols_results.params[i],
                't-value':ols_results.tvalues[i],
                'Std_Error':ols_results.bse[i],
                'F-stat':ols_results.fvalue,
                'p-val(F-stat)': ols_results.f_pvalue,
                'DW-stat':durbin_watson(ols_results.resid),
                'Flag_estimate':int(flag_est[0]),
                'Flag_p-value':int(flag_p_value[0])
            })


        return list_model_results
    else:
        return None



def build_custom_regression(X, y, list_of_vars):

    # Add constant (for sm)
    X = sm.add_constant(X)

    # Train/Validation Split
    X_train, X_val, y_train, y_val = create_validation_set(X, y)

    # Build Model only on training set
    model = sm.OLS(y_train, X_train)
    ols_results = model.fit()
    print(ols_results.summary())

    # ===============
    # Model Selection
    # ===============

    vif_info = get_vif(X)
    coeff = ols_results.params
    p_value_limit = model_config_params['p_value_limit'] 
    r_square_limit = model_config_params['r_square_limit'] 
    vif_limit = model_config_params['vif_limit'] 
    
    media_vars = list_of_vars['media']
    comp_vars = list_of_vars['competitor']
    other_vars = list_of_vars['others']

    accept_model = False
    # Check if competitior variables are present, if yes, create selection criteria
    if comp_vars:
        if all(ols_results.params[media_vars] > 0) and \
            all(ols_results.params[comp_vars] < 0) and \
            max(ols_results.pvalues) < p_value_limit and \
            round(ols_results.rsquared,3) > r_square_limit and \
            all(vif_info) < vif_limit :
            accept_model = True
        else:
            print('\n The Model did not satisfy conditions for model selection \n')
    else:
        if all(ols_results.params[media_vars] > 0) and \
            max(ols_results.pvalues) < p_value_limit and \
            round(ols_results.rsquared,3) > r_square_limit and \
            all(vif_info) < vif_limit :
            accept_model = True
        else:
            print('\n The Model did not satisfy conditions for model selection \n')

    # Only proceed further if the model is selected
    # In case of a manual run, if the model selection fails,
    # Quit with some meanigful error message!

    # Training data results
    y_pred_train, model_results_train = get_error_metrics(X_train, y_train, ols_results)
    # Validation data results

    # Predict on train and validation set and evaluate metrics
    y_pred_val = ols_results.predict(X_val)
    exp_var, r2, mae, mse, rmse, mape = regression_results(y_val, y_pred_val)

    validation_results = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    all_results = {'Train': [y_pred_train, model_results_train],\
                   'Validate': [y_pred_val, validation_results]}

    return all_results


def process_data(feature, df_current_asl, previous_features, previous_df):

    
    # Join current feature series with previous series!
    # Append and concat (easiest)
    current_series = df_current_asl[feature].copy()
    if previous_features:
        previous_current = previous_df.copy()
        previous_current.append(current_series.to_frame())
        X_current = pd.concat(previous_current, axis=1)
    else:
        X_current = current_series.to_frame()#.reset_index()
    
    # Build model 
    #current_model, current_model_output = build_regression(X_current, y_target, model_counter)
    current_model_output = build_stepwise_regression(X_current, y_target)

    return current_model_output


def parallel_processing(all_features, df_current_asl, previous_features, previous_df):

    num_cores = 8  # Number of CPU cores to use for parallel processing
    with tqdm_joblib(tqdm(desc="Building Regression Models", total=10)) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(process_data)(feature, df_current_asl, previous_features, previous_df) for feature in all_features)

    return results

def forward_selection_media_variables(media_variables):
    
    # Output Path
    output_path = model_config_params['output_path']

    # Read Dict - Variables and Coefficients
    list_var = list(media_variables.keys())
    list_val = list(media_variables.values())
    n_media_features = len(list_var)

    # Get index of 'Current' variable
    #current_var_index = list_val.index('Current')

    # Get index of None in 'forward_selection_variables'
    current_none_index = list_val.index(None)

    # If the None index is 0, that means it is the first run of
    # forward selection and there are no previous features
    # If it is non-zero, then simply take the  previous index 
    # as the previous feature
    feature_to_add = list_var[current_none_index]

    print()
    print('Adding feature to the model: ', feature_to_add) 
    print()

    if current_none_index == 0:
        print('This is the first run of forward selection!')
        previous_features = []
        previous_df = []
    else:
        previous_models   =  list_val[:current_none_index]
        previous_features =  list_var[:current_none_index]
        previous_iteration_feature = list_var[current_none_index-1]
        best_model_previous_run = media_variables[previous_iteration_feature]
        for j in range(1, len(previous_models)):
            if best_model_previous_run == 'Skip':
                previous_iteration_feature =  list_var[current_none_index-j-1]
                best_model_previous_run = media_variables[previous_iteration_feature]
                continue
            else:
                break

        # Read previous iteration result dataframe (with model output)
        df_previous_iteration_results = pd.read_csv(os.path.join(output_path, previous_iteration_feature + '.csv'))
        # Check if best model from previous run actually exists (could be a user error)
        if best_model_previous_run in df_previous_iteration_results['Model'].unique():
            pass
        else:
            print('ERROR')
            print('It seems that the best model number from previous run does not exist in the provided file')
            print('Check input for forward selection and re-run!')
            sys.exit()
        # Get best previous model
        df_previous_iteration_best_model = df_previous_iteration_results[df_previous_iteration_results['Model'] == best_model_previous_run] 
        best_variables_previous_iteration = list(df_previous_iteration_best_model['Variable'].values)

        # You need best ASL transformed variables columns from the previous run
        # You can either read them from the file, or better yet, just compute
        # them right here (anyway beneficial to have ASL code here, will need later!)
        previous_df = []
        for variable in best_variables_previous_iteration:
            if variable not in ['const', 'constant']:
                current_variable = '_'.join(variable.split('_')[:-5])
                current_X = df_input[current_variable].copy()
                asl_params = [float(x) for x in variable.split('_')[-4:]]
                current_X_asl = apply_asl_sequentially(current_X, asl_params)
                current_X_asl = current_X_asl.rename(variable)
                previous_df.append(current_X_asl)
            else:
                pass

    # Read ASL dump (or compute on-the-fly) - Precomputed ASL columns
    asl_path = model_config_params['asl_dump_path']
    df_current_asl = pd.read_csv(asl_path + feature_to_add + '.csv')

    # TODO: This needs to be fixed, atm you have to drop first and last column
    asl_features_current = list(df_current_asl.columns.values)[1:-1]

    # Modeling - Joblib
    all_outputs = parallel_processing(asl_features_current, df_current_asl, previous_features, previous_df)

    # Process output
    all_model_outputs = []
    model_counter = 1
    for output in all_outputs:
        if output is not None:
            new_output = []
            for out in output:
                model_dict = {'Model': model_counter}
                model_dict.update(out)
                new_output.append(model_dict)
            all_model_outputs.append(new_output)
            model_counter += 1
        else:
            pass

    # Save results to DF. Add contribution too!
    df_current_media = pd.DataFrame(list(chain.from_iterable(all_model_outputs)))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        pass
    df_current_media.to_csv(os.path.join(output_path, feature_to_add + '.csv'))


    print('All models done, out of for loop')


    return


def custom_variables_model(list_of_vars, output_path):

    media_vars = list_of_vars['media']
    comp_vars = list_of_vars['competitor']
    other_vars = list_of_vars['others']

    # Compute ASL transformations for media variables
    media_df = []
    for var in media_vars + comp_vars:
        current_variable = '_'.join(var.split('_')[:-5])
        current_X = df_input[current_variable].copy()
        asl_params = [float(x) for x in var.split('_')[-4:]]
        current_X_asl = apply_asl_sequentially(current_X, asl_params)
        current_X_asl = current_X_asl.rename(var)
        media_df.append(current_X_asl)

    # Read other variables from df and concat to create full independent df
    # Which will go in the model (X)
    if other_vars:
        other_df = df_input[other_vars]
        media_df.append(other_df)
    else:
        pass

    # Input DataFrame for all independent variables (media + comp + others)
    X_input = pd.concat(media_df, axis=1)

    # Model building and selection
    dict_results = build_custom_regression(X_input, y_target, list_of_vars)

    # Predict on training and validation
    X_train, X_val, y_train, y_val = create_validation_set(X_input, y_target)
    y_pred_train, results_train = dict_results['Train'][0], dict_results['Train'][1]
    y_pred_val, results_val = dict_results['Validate'][0], dict_results['Validate'][1]

    # Write results to file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        pass

    pd.DataFrame({'Orig': y_train, 'Pred': y_pred_train}).to_csv(os.path.join(output_path,'actual_vs_pred_train.csv'))
    pd.DataFrame({'Orig': y_val, 'Pred': y_pred_val}).to_csv(os.path.join(output_path,'actual_vs_pred_validation.csv'))
    pd.DataFrame(results_train).to_csv(os.path.join(output_path,'model_summary_train.csv'))

    with open(os.path.join(output_path,'model_validation.txt'), 'w') as file:
        file.write(json.dumps(results_val))

    return
