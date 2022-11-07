import json
import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from univariate_forecast_daily.util.metrics import calculate_modified_metrics
from univariate_forecast_daily.util.model import get_best_models

AVERAGE_DATASET_SET_LIST = [
    [
        'lk',
        (
            'boralasgamuwa_uc_2012-2018',
            'moratuwa_mc_2014-2018',
            'dehiwala_mc_2012-2018'
        )
    ],
    [
        'ballarat',
        (
            'open_source_ballarat_daily_waste_2000_jul_2015_mar',
        )
    ],
    [
        'austin',
        (
            'open_source_austin_daily_waste_2003_jan_2021_jul',
        )
    ],
    [
        'all',
        (
            'boralasgamuwa_uc_2012-2018',
            'moratuwa_mc_2014-2018',
            'dehiwala_mc_2012-2018',
            'open_source_ballarat_daily_waste_2000_jul_2015_mar',
            'open_source_austin_daily_waste_2003_jan_2021_jul'
        )
    ]
]

SINGLE_MODEL_NAME_LIST = [
    '[Single] Linear Regression',
    '[Single] Auto ARIMA',
    '[Single] Random Forest',
    '[Single] Light GBM',
    '[Single] Prophet',
    '[Single] LSTM',
    '[Single] TCN',
    '[Single] Transformer',
    '[Single] N-BEATS'
]

MULTI_MODEL_NAME_LIST = [
    '[Multi] Linear Regression',
    '[Multi] Auto ARIMA',
    '[Multi] Random Forest',
    '[Multi] Light GBM',
    '[Multi] Prophet',
    '[Multi] LSTM',
    '[Multi] TCN',
    '[Multi] Transformer',
    '[Multi] N-BEATS'
]

REPORT_OUTPUT_DIR_PATH_PATTERN = '../../tmp/univariate_forecast_daily/reports/average_bar_plot/{}'

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['figure.figsize'] = (14, 4)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.rcParams['legend.fontsize'] = 17
    plt.rcParams['legend.title_fontsize'] = 17
    plt.rcParams.update({'font.size': 17})

    for average_dataset_set_name, average_dataset_set in AVERAGE_DATASET_SET_LIST:
        print('#### {}'.format(average_dataset_set_name))

        rmse_dict = {}
        mae_dict = {}
        mape_dict = {}
        training_time_dict = {}
        predicting_time_dict = {}

        report_output_dir_path = REPORT_OUTPUT_DIR_PATH_PATTERN.format(average_dataset_set_name)

        # Create report dir
        if not os.path.exists(report_output_dir_path):
            os.makedirs(report_output_dir_path)

        model_detail_list = []
        for dataset_name in average_dataset_set:
            training_output_dir_path = f'../../tmp/univariate_forecast_daily/{dataset_name}'

            calculate_modified_metrics(dataset_name)
            chosen_model_list = get_best_models(training_output_dir_path)

            for dir_name, model_name in chosen_model_list:
                with open('{}/{}/modified_summary.json'.format(training_output_dir_path, dir_name)) as json_file:
                    summary = json.load(json_file)

                model_detail_list.append([
                    summary['rmse'],
                    summary['mae'],
                    summary['mape'],
                    summary['training_time'],
                    summary['predicting_time'],
                    model_name
                ])

        for rmse_val, mae_val, mape_val, training_time, predicting_time, model_name in model_detail_list:
            if model_name not in rmse_dict:
                rmse_dict[model_name] = []
                mae_dict[model_name] = []
                mape_dict[model_name] = []
                training_time_dict[model_name] = []
                predicting_time_dict[model_name] = []

            rmse_dict[model_name].append(rmse_val)
            mae_dict[model_name].append(mae_val)
            mape_dict[model_name].append(mape_val)
            training_time_dict[model_name].append(training_time)
            predicting_time_dict[model_name].append(predicting_time)


        def plot_average_bar_chart(single_model_average_list, multi_model_average_list, plot_name, y_label, y_lim):
            labels = [model_name[9:] for model_name in SINGLE_MODEL_NAME_LIST]
            temp_labels = []
            for label in labels:
                if label == 'Linear Regression':
                    label = 'Linear\nRegression'
                elif label == 'Auto ARIMA':
                    label = 'Auto\nARIMA'
                elif label == 'Random Forest':
                    label = 'Random\nForest'
                elif label == 'Light GBM':
                    label = 'Light\nGBM'

                temp_labels.append(label)
            labels = temp_labels

            x = np.arange(len(labels))  # the label locations
            width = 0.3  # the width of the bars

            fig, ax = plt.subplots()
            means1 = ax.bar(x - width / 2 - 0.01, single_model_average_list, width, label='Single-model approach',
                            color='#B71C1C', edgecolor='white', hatch='/')
            means2 = ax.bar(x + width / 2 + 0.01, multi_model_average_list, width, label='Multi-model approach',
                            color='#1B5E20', edgecolor='white', hatch='.')

            ax.set_ylabel(y_label)
            ax.set_xticks(x, labels, color='black')
            ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(0.1, 1.2))

            if y_lim:
                ax.set_ylim([0, y_lim])

            ax.bar_label(means1, padding=5, rotation=90)
            ax.bar_label(means2, padding=5, rotation=90)

            fig.tight_layout()

            plt.savefig(f'../../tmp/univariate_forecast_daily/reports/average_bar_plot/{average_dataset_set_name}/{plot_name}.eps', format='eps')


        single_model_average_mape_list = []
        single_model_average_training_time_list = []
        single_model_average_predicting_time_list = []
        for model_name in SINGLE_MODEL_NAME_LIST:
            average_rmse = mean(rmse_dict[model_name])
            average_mae = mean(mae_dict[model_name])
            average_mape = mean(mape_dict[model_name])
            average_training_time = mean(training_time_dict[model_name])
            average_predicting_time = mean(predicting_time_dict[model_name])

            print('& {} & {:.2f} & {:.2f} & {:.2f}\% \\\\'.format(model_name[9:], average_rmse, average_mae,
                                                                  average_mape))

            single_model_average_mape_list.append(average_mape)
            single_model_average_training_time_list.append(average_training_time)
            single_model_average_predicting_time_list.append(average_predicting_time)

        multi_model_average_mape_list = []
        multi_model_average_training_time_list = []
        multi_model_average_predicting_time_list = []
        for model_name in MULTI_MODEL_NAME_LIST:
            average_rmse = mean(rmse_dict[model_name])
            average_mae = mean(mae_dict[model_name])
            average_mape = mean(mape_dict[model_name])
            average_training_time = mean(training_time_dict[model_name])
            average_predicting_time = mean(predicting_time_dict[model_name])

            print('& {} & {:.2f} & {:.2f} & {:.2f}\% \\\\'.format(model_name[8:], average_rmse, average_mae,
                                                                  average_mape))

            multi_model_average_mape_list.append(average_mape)
            multi_model_average_training_time_list.append(average_training_time)
            multi_model_average_predicting_time_list.append(average_predicting_time)

        single_model_average_mape_list = [round(elem, 1) for elem in single_model_average_mape_list]
        multi_model_average_mape_list = [round(elem, 1) for elem in multi_model_average_mape_list]

        single_model_average_training_time_list = [round(elem, 1) for elem in single_model_average_training_time_list]
        multi_model_average_training_time_list = [round(elem, 1) for elem in multi_model_average_training_time_list]

        single_model_average_predicting_time_list = [round(elem, 1) for elem in
                                                     single_model_average_predicting_time_list]
        multi_model_average_predicting_time_list = [round(elem, 1) for elem in multi_model_average_predicting_time_list]

        y_label = 'MAPE (%)'
        if len(average_dataset_set) > 1:
            y_label = 'Average MAPE (%)'

        plot_average_bar_chart(
            single_model_average_mape_list,
            multi_model_average_mape_list,
            'average_mape',
            y_label='Average MAPE (%)' if len(average_dataset_set) > 1 else 'MAPE (%)',
            y_lim=100
        )
        plot_average_bar_chart(
            single_model_average_training_time_list,
            multi_model_average_training_time_list,
            'average_training_time',
            y_label='Average time (s)' if len(average_dataset_set) > 1 else 'Time (s)',
            y_lim=None
        )
        plot_average_bar_chart(
            single_model_average_predicting_time_list,
            multi_model_average_predicting_time_list,
            'average_predicting_time',
            y_label='Average time (s)' if len(average_dataset_set) > 1 else 'Time (s)',
            y_lim=None
        )
