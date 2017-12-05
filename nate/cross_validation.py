import pdb

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

def append_bias(x, axis=1, val=1):
    x = pd.DataFrame(data=x)
    x['bias'] = [1] * x.shape[0]
    x = np.array(x)
    return x
    #return np.insert(x, x.shape[axis], [val]*x.shape[0], axis=axis)

def data_from_file(filepath, is_svmlight=False):
    if is_svmlight:
        return load_svmlight_file(filepath)
    return pd.read_csv(filepath)

def get_xy(data, is_svmlight=False, y_col=None, features_to_drop=[]):
    if is_svmlight:
        return data[0].todense(), data[1]
    for feature_to_drop in features_to_drop:
        data = data.drop(feature_to_drop, axis=1)
    y = data[y_col]
    y = y.astype(int)
    y = y.replace(to_replace=0, value=-1)
    y = np.reshape(y, (len(y), 1))
    data.drop(y_col, axis=1)
    return data, y

def add_any_missing_features_cols(data, num_features):
    if num_features is not None and data.shape[1] < num_features:
        return np.concatenate([data, np.zeros([data.shape[0], num_features - data.shape[1]])], axis=1)
    return data

def cross_val(folds, cv_filepath, num_features=None, add_bias=True, leave_out_none=False, is_svmlight=False, y_col=None, features_to_drop=[]):
    eval_x, eval_y = None, None
    for leave_out in range(folds):
        x, y = None, None
        for cv in range(folds):
            data = data_from_file(cv_filepath.format(cv=cv), is_svmlight=is_svmlight)
            # save one fold for evaluation data
            if cv == leave_out and not leave_out_none:
                eval_x, eval_y = get_xy(data, is_svmlight=is_svmlight, y_col=y_col, features_to_drop=features_to_drop)
                # it we're missing any features columns, add them as zeros
                eval_x = add_any_missing_features_cols(eval_x, num_features)
                # add bias term to instances
                if add_bias:
                    eval_x = append_bias(eval_x)
                continue
            # if this is the first training fold, just read in the data
            if x is None and y is None:
                x, y = get_xy(data, is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
                # it we're missing any features columns, add them as zeros
                x = add_any_missing_features_cols(x, num_features)
            else:
                temp_x, temp_y = get_xy(data, is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
                # it we're missing any features columns, add them as zeros
                temp_x = add_any_missing_features_cols(temp_x, num_features)
                # if this isn't the first fold, tack on to training data
                x = np.concatenate((x, temp_x))
                y = np.concatenate((y, temp_y))
        # add bias term to instances
        if add_bias:
            x = append_bias(x)
        yield x, y, eval_x, eval_y
