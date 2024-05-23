from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import bootstrap_method
import bootstrap_analysis

def process_data(df):
    # remove all columns that have no data on the proteins and convert 'NaN' values into None
    clean_df = df.copy()
    clean_df.replace({np.nan: None}, inplace=True)
    clean_df['Biomarker_1'] = np.log(df['Biomarker_1'])
    clean_df['Biomarker_2'] = np.log(df['Biomarker_2'])
    clean_df['Biomarker_3'] = np.log(df['Biomarker_3']) * -1
    return clean_df


def data_time_split(df, target):
    data = defaultdict(list)
    time = defaultdict(list)
    for index, row in df.iterrows():
        data[row['ID']].append(str(round(float(row[target]), 3)))
        time[row['ID']].append(row['Time'])

    return data, time

def obtain_best_model(call, metric):

    best_trans_mat_0 = [[call[0][0][metric], call[0][1][metric], call[0][2][metric]]]
    best_trans_mat_1 = [[call[0][3][metric], call[0][4][metric], call[0][5][metric]]]
    best_trans_mat_2 = [[call[0][6][metric], call[0][7][metric], call[0][8][metric]]]

    best_shape_params = list((call[1][0][metric],
                              call[1][1][metric],
                              call[1][2][metric]))

    best_scale_params = list((call[2][0][metric],
                              call[2][1][metric],
                              call[2][2][metric]))

    best_init_probs = list((call[4][0][metric],
                            call[4][1][metric],
                            call[4][2][metric]))

    best_gamma_means = list((call[3][0][metric],
                             call[3][1][metric],
                             call[3][2][metric]))

    x = np.linspace(0, 10, 10)

    y1 = stats.gamma.pdf(x, a=best_shape_params[0], scale=best_scale_params[0])
    y2 = stats.gamma.pdf(x, a=best_shape_params[1], scale=best_scale_params[1])
    y3 = stats.gamma.pdf(x, a=best_shape_params[2], scale=best_scale_params[2])

    # add lines for each distribution
    plt.plot(x, y1, label='State 0: Healthy')
    plt.plot(x, y2, label='State 1: Symptomatic')
    plt.plot(x, y3, label='State 2: Critical')

    # add legend
    plt.legend()

    # display plot
    plt.show()

    print("Best Transition Matrix\n")
    print(best_trans_mat_0)
    print(best_trans_mat_1)
    print(best_trans_mat_2)
    print('\n')

    print("Best Shape Parameters\n")
    print(np.concatenate(best_shape_params))
    print('\n')

    print('Best Scale Parameters\n')
    print(np.concatenate(best_scale_params))
    print('\n')

    print('Best Initial Probabilities\n')
    print(np.concatenate(best_init_probs))
    print('\n')

    print('Best Gamma Distribution Means\n')
    print(np.concatenate(best_gamma_means))
    print('\n')

    best_trans_mat = [best_trans_mat_0, best_trans_mat_1, best_trans_mat_2]
    best_parameters = [best_trans_mat, best_shape_params, best_scale_params, best_init_probs, best_gamma_means]
    return (best_parameters)

def visualize_confidence_intervals(call):
    print('Obtiaining the Confidence Intervals for Biomarker 1\n')
    for i in range(5):
        print(call[0][1][i])
        print('\n')

    print('Obtiaining the Confidence Intervals for Biomarker 2\n')
    for i in range(5):
        print(call[1][1][i])
        print('\n')

    print('Obtiaining the Confidence Intervals for Biomarker 3\n')
    for i in range(5):
        print(call[2][1][i])
        print('\n')

    return 0

def visualize_model_scores(call):
    plt.bar(list(range(len(call[0][0][5]))), np.array(call[0][0][5]).flatten(), color='red')
    plt.title('Modeled Using Biomarker 1')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('AIC Score')
    plt.show()
    plt.bar(list(range(len(call[0][0][6]))), np.array(call[0][0][6]).flatten(), color='pink')
    plt.title('Modeled Using Biomarker1')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('BIC Score')
    plt.show()
    plt.bar(list(range(len(call[1][0][5]))), np.array(call[1][0][5]).flatten(), color='blue')
    plt.title('Modeled Using Biomarker 2')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('AIC Score')
    plt.show()
    plt.bar(list(range(len(call[1][0][6]))), np.array(call[1][0][6]).flatten(), color='lightblue')
    plt.title('Modeled Using Biomarker 2')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('BIC Score')
    plt.show()
    plt.bar(list(range(len(call[2][0][5]))), np.array(call[2][0][5]).flatten(), color='green')
    plt.title('Modeled Using Biomarker 3')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('AIC Score')
    plt.show()
    plt.bar(list(range(len(call[2][0][6]))), np.array(call[2][0][6]).flatten(), color='lightgreen')
    plt.title('Modeled Using Biomarker 3')
    plt.xlabel('Bootstrap Iteration Number')
    plt.ylabel('BIC Score')
    plt.show()

    print('-----------------------------------------')
    print('-- MODEL RESULTS FOR BIOMARKER 1 --')
    print(f'Best AIC Score: {min(call[0][0][5])}')
    print(f'Mean AIC Score: {sum(call[0][0][5]) / len(call[0][0][5])}')
    print(f'Difference: {(sum(call[0][0][5]) / len(call[1][0][5])) - (min(call[0][0][5]))}\n')
    print(f'Best BIC Score: {min(call[0][0][6])}')
    print(f'Mean BIC Score: {sum(call[0][0][6]) / len(call[0][0][6])}')
    print(f'Difference: {(sum(call[0][0][6]) / len(call[0][0][6])) - (min(call[0][0][6]))}\n')
    print('-----------------------------------------')

    print('-- MODEL RESULTS FOR BIOMARKER 2 --')
    print(f'Best AIC Score: {min(call[1][0][5])}')
    print(f'Mean AIC Score: {sum(call[1][0][5]) / len(call[1][0][5])}')
    print(f'Difference: {(sum(call[1][0][5]) / len(call[1][0][5])) - (min(call[1][0][5]))}\n')
    print(f'Best BIC Score: {min(call[1][0][6])}')
    print(f'Mean BIC Score: {sum(call[1][0][6]) / len(call[1][0][6])}')
    print(f'Difference: {(sum(call[1][0][6]) / len(call[1][0][5])) - (min(call[1][0][6]))}\n')
    print('-----------------------------------------')

    print('-- MODEL RESULTS FOR BIOMARKER 3 --')
    print(f'Best AIC Score: {min(call[2][0][5])}')
    print(f'Mean AIC Score: {sum(call[2][0][5]) / len(call[2][0][5])}')
    print(f'Difference: {(sum(call[2][0][5]) / len(call[2][0][5])) - (min(call[2][0][5]))}\n')
    print(f'Best BIC Score: {min(call[2][0][6])}')
    print(f'Mean BIC Score: {sum(call[2][0][6]) / len(call[2][0][6])}')
    print(f'Difference: {(sum(call[2][0][6]) / len(call[2][0][5])) - (min(call[2][0][6]))}\n')

    return 0

def main():
    df = pd.read_csv('synthetic_data.csv')
    file = process_data(df)

    data, time_intervals = data_time_split(file, 'Biomarker_1')
    results_1 = bootstrap_method.bootstrap(data, time_intervals)
    df_results_1 = bootstrap_analysis.analyize(results_1)
    print('First Biomarker Done!')

    data, time_intervals = data_time_split(file, 'Biomarker_2')
    results_2 = bootstrap_method.bootstrap(data, time_intervals)
    df_results_2 = bootstrap_analysis.analyize(results_2)
    print('Second Biomarker Done!')

    data, time_intervals = data_time_split(file, 'Biomarker_3')
    results_3 = bootstrap_method.bootstrap(data, time_intervals)
    df_results_3 = bootstrap_analysis.analyize(results_3)
    print('Third Biomarker Done!')

    outcome = [[results_1, df_results_1], [results_2, df_results_2], [results_3, df_results_3]]
    visualize_model_scores(outcome)
    visualize_confidence_intervals(outcome)
    obtain_best_model(outcome[1][0], outcome[1][0][5].index(min(outcome[1][0][5])))

    return outcome

if (__name__ == '__main__'):
    main()

