import statsmodels.tsa.api as tsa

def geometric_adstock(col, lamda):
    """
    lamda = retention
    """
    return tsa.filters.recursive_filter(col, lamda)


def s_curve_saturation(col, alpha, gamma):
    """
    x = array
    alpha = shape
    gamma = steepness/inflection
    """
    return (col**alpha) / (col ** alpha + gamma ** alpha)


def lag(col, periods):
    periods = int(periods)
    return col.shift(periods).fillna(0)


def apply_asl_sequentially(col, params):
    """
    params = [lamda, alpha, gamma, periods]
    """
 
    # Unzip params
    retention = params[0]
    shape = params[1]
    steepness = params[2]
    lag_period = params[3]

    # Apply transformations
    adstocked_col = geometric_adstock(col, retention)
    saturated_col = adstocked_col * s_curve_saturation(adstocked_col, shape, steepness)
    lag_col       = lag(saturated_col, lag_period)

    return lag_col
