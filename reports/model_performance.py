import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from univariate_forecast_daily.util.metrics import calculate_modified_metrics
from univariate_forecast_daily.util.model import get_best_models

DATASET_NAME_LIST = [
    'boralasgamuwa_uc_2012-2018',
    'moratuwa_mc_2014-2018',
    'dehiwala_mc_2012-2018',
    'open_source_ballarat_daily_waste_2000_jul_2015_mar',
    'open_source_austin_daily_waste_2003_jan_2021_jul'
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

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['figure.figsize'] = (14, 7)

    for dataset_name in DATASET_NAME_LIST:
        training_output_dir_path = f'../../tmp/univariate_forecast_daily/{dataset_name}'
        report_output_dir_path = f'../../tmp/univariate_forecast_daily/reports/model_performance/{dataset_name}'

        calculate_modified_metrics(dataset_name)
        chosen_model_list = get_best_models(training_output_dir_path)

        # Create report dir
        if not os.path.exists(report_output_dir_path):
            os.makedirs(report_output_dir_path)

        rmse_fig = plt.figure(1)
        mape_fig = plt.figure(2)
        mape_bar_fig = plt.figure(3)

        rmse_ax = rmse_fig.add_subplot(111)
        mape_ax = mape_fig.add_subplot(111)
        mape_bar_ax = mape_bar_fig.add_subplot(111)

        # Calculate RMSE, MAE, MAPE
        model_detail_list = []
        model_mape_dict = {}
        for dir_name, model_name in chosen_model_list:
            output_df = pd.read_csv(f'{training_output_dir_path}/{dir_name}/modified_output.csv')

            with open(f'{training_output_dir_path}/{dir_name}/params.json') as json_file:
                params = json.load(json_file)

            with open(f'{training_output_dir_path}/{dir_name}/modified_summary.json') as json_file:
                summary = json.load(json_file)

            line_label = '{} - {:,.2f}, {:,.2f},  {:,.2f}%'.format(
                model_name,
                summary['rmse'],
                summary['mae'],
                summary['mape']
            )

            model_detail_list.append([
                summary['rmse'],
                summary['mae'],
                summary['mape'],
                model_name,
                line_label,
                dir_name,
                #             output_df.groupby(np.arange(len(output_df)) // 25)['rmse'].mean().reset_index(),
                #             output_df.groupby(np.arange(len(output_df)) // 25)['mape'].mean().reset_index(),
                output_df['rmse'],
                output_df['mape'],
                params
            ])

            model_mape_dict[model_name] = summary['mape']

        # Sort by rmse_val
        model_detail_list = sorted(model_detail_list, key=lambda x: x[2])

        # Print details
        leaderboard_list = []
        for rmse_val, mae_val, mape_val, model_name, line_label, dir_name, rmse_series, mape_series, params in model_detail_list:
            rmse_series.plot(ax=rmse_ax, label=line_label)
            mape_series.plot(ax=mape_ax, label=line_label)

            leaderboard_list.append('')
            leaderboard_list.append(line_label)
            leaderboard_list.append(dir_name)

        with open(f'{report_output_dir_path}/leaderboard.txt', 'w') as f:
            for line in leaderboard_list:
                f.write('{}\n'.format(line))

        plt.figure(1)
        plt.title('RMSE')
        plt.legend()
        plt.savefig('{}/rmse.png'.format(report_output_dir_path))

        plt.figure(2)
        plt.ylim(0, 100)
        plt.title('MAPE')
        plt.legend()
        plt.savefig('{}/mape.png'.format(report_output_dir_path))

        single_model_mape_list = []
        for model_name in SINGLE_MODEL_NAME_LIST:
            single_model_mape_list.append(model_mape_dict[model_name])

        multi_model_mape_list = []
        for model_name in MULTI_MODEL_NAME_LIST:
            multi_model_mape_list.append(model_mape_dict[model_name])

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
        means1 = ax.bar(x - width / 2 - 0.01, single_model_mape_list, width, label='Single-model approach',
                        color='#B71C1C', edgecolor='white', hatch='/')
        means2 = ax.bar(x + width / 2 + 0.01, multi_model_mape_list, width, label='Multi-model approach',
                        color='#1B5E20', edgecolor='white', hatch='.')

        ax.set_ylabel('MAPE')
        ax.set_xticks(x, labels, color='black')
        ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(0.7, 1.2))

        ax.bar_label(means1, padding=5, rotation=90)
        ax.bar_label(means2, padding=5, rotation=90)

        fig.tight_layout()

        plt.savefig('{}/mape_bar.png'.format(report_output_dir_path))
