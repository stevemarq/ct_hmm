import numpy as np
import pandas as pd

def analyize(list_results):
    # we want confidence_interval => 0.95
    a = 0.05
    lower_percentile = (a / 2) * 100
    upper_percentile = (1 - a + (a / 2)) * 100

    # lists to store the data bounded by the percentile
    trans_lower = [np.percentile(trans, lower_percentile) for trans in list_results[0]]
    trans_upper = [np.percentile(trans, upper_percentile) for trans in list_results[0]]

    shape_lower = [np.percentile(shape, lower_percentile) for shape in list_results[1]]
    shape_upper = [np.percentile(shape, upper_percentile) for shape in list_results[1]]
    scale_lower = [np.percentile(scale, lower_percentile) for scale in list_results[2]]
    scale_upper = [np.percentile(scale, upper_percentile) for scale in list_results[2]]

    mean_lower = [np.percentile(mean, lower_percentile) for mean in list_results[3]]
    mean_upper = [np.percentile(mean, upper_percentile) for mean in list_results[3]]

    init_lower = [max(0, np.percentile(init_prob, lower_percentile)) for init_prob in list_results[4]]
    init_upper = [min(1, np.percentile(init_prob, upper_percentile)) for init_prob in list_results[4]]


    # for a 3 state model
    trans_mat_parameters = ['trans_00', 'trans_01', 'trans_02',
                            'trans_10', 'trans_11', 'trans_12',
                            'trans_20', 'trans_12', 'trans_22']
    trans_df = pd.DataFrame({'parameters': trans_mat_parameters, 'lower': trans_lower, 'upper': trans_upper})

    shape_parameters = ['shape_1', 'shape_2', 'shape_3']
    scale_parameters = ['scale_1', 'scale_2', 'scale_3']
    shape_df = pd.DataFrame({'parameters': shape_parameters, 'lower': shape_lower, 'upper': shape_upper})

    scale_df = pd.DataFrame({'parameters': scale_parameters, 'lower': scale_lower, 'upper': scale_upper})

    init_prob_parameters = ['init_prob_1', 'init_prob_2', 'init_prob_3']
    ip_df = pd.DataFrame({'parameters': init_prob_parameters, 'lower': init_lower, 'upper': init_upper})


    mean_parameters = ['mean_1', 'mean_2', 'mean_3']
    mean_df = pd.DataFrame({'parameters': mean_parameters, 'lower': mean_lower, 'upper': mean_upper})


    # for a 2 state model
#     trans_mat_parameters = ['trans_00', 'trans_01',
#                             'trans_10', 'trans_11']
#     trans_df = pd.DataFrame({'parameters': trans_mat_parameters, 'lower': trans_lower, 'upper': trans_upper})

#     shape_parameters = ['shape_1', 'shape_2']
#     scale_parameters = ['scale_1', 'scale_2']
#     shape_df = pd.DataFrame({'parameters': shape_parameters, 'lower': shape_lower, 'upper': shape_upper})

#     scale_df = pd.DataFrame({'parameters': scale_parameters, 'lower': scale_lower, 'upper': scale_upper})

#     init_prob_parameters = ['init_prob_1', 'init_prob_2']
#     ip_df = pd.DataFrame({'parameters': init_prob_parameters, 'lower': init_lower, 'upper': init_upper})


#     mean_parameters = ['mean_1', 'mean_2']
#     mean_df = pd.DataFrame({'parameters': mean_parameters, 'lower': mean_lower, 'upper': mean_upper})

    dataframes = [trans_df, shape_df, scale_df, ip_df, mean_df]
    return dataframes