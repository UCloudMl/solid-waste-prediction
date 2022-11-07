import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from univariate_forecast_daily.util.model import get_best_models

DATASET_NAME_LIST = [
    ['boralasgamuwa_uc_2012-2018', 'Boralesgamuwa'],
    ['dehiwala_mc_2012-2018', 'Dehiwala'],
    ['moratuwa_mc_2014-2018', 'Moratuwa'],
    ['open_source_austin_daily_waste_2003_jan_2021_jul', 'Austin'],
    ['open_source_ballarat_daily_waste_2000_jul_2015_mar', 'Ballarat']
]

PREFIXES = [
    ['daily_lr_weekdays', '[Single] Linear Regression'],
    ['daily_arima_weekdays', '[Single] Auto ARIMA'],
    ['daily_rf_weekdays', '[Single] Random Forest'],
    ['daily_light_gbm_weekdays', '[Single] Light GBM'],
    ['daily_prophet_weekdays', '[Single] Prophet'],
    ['daily_lstm_weekdays', '[Single] LSTM'],
    ['daily_tcn_weekdays', '[Single] TCN'],
    ['daily_transformer_weekdays', '[Single] Transformer'],
    ['daily_n_beats_weekdays', '[Single] N-BEATS'],

    # MULTI ##############################################################
    ['daily_lr_multi_weekdays', '[Multi] Linear Regression'],
    ['daily_arima_multi_weekdays', '[Multi] Auto ARIMA'],
    ['daily_rf_multi_weekdays', '[Multi] Random Forest'],
    ['daily_light_gbm_multi_weekdays', '[Multi] Light GBM'],
    ['daily_lstm_multi_weekdays', '[Multi] LSTM'],
    ['daily_tcn_multi_weekdays', '[Multi] TCN'],
    ['daily_transformer_multi_weekdays', '[Multi] Transformer'],
    ['daily_n_beats_multi_weekdays', '[Multi] N-BEATS']
]

N_DAYS = 7

RESULT_OUTPUT_DIR_PATH_FORMAT = '../../tmp/univariate_forecast_daily/{}'
REPORT_OUTPUT_DIR_PATH = '../../tmp/univariate_forecast_daily/reports/area_mape_plots'


def plot_stuff(chosen_model_list, result_output_dir_path, dataset_name, formatted_name):
    # Choose best, worst models
    best_model = chosen_model_list[0]
    average_model = chosen_model_list[int(len(chosen_model_list) / 2)]
    worst_model = chosen_model_list[-1]

    # [model_type, marker, color, dir_name, model_name]
    plot_model_list = [
        [
            'Best',
            'D',
            '#388E3C',
            best_model[0],
            best_model[1]
        ],
        [
            'Average',
            's',
            '#1976D2',
            average_model[0],
            average_model[1]
        ],
        [
            'Worst',
            'o',
            '#D32F2F',
            worst_model[0],
            worst_model[1]
        ]
    ]

    fig, ax = plt.subplots()

    for model_type, marker, color, dir_name, model_name in plot_model_list:
        output_df = pd.read_csv('{}/{}/modified_output.csv'.format(result_output_dir_path, dir_name))

        line_label = '{}'.format(model_name)

        output_df['mape'][:N_DAYS].plot(
            label=line_label,
            color=color,
            marker=marker,
            markerfacecolor='none',
            linewidth=2,
            markersize=8
        )

    # plt.title('RMSE - {}'.format(formatted_name))
    plt.xticks(np.arange(7), np.arange(1, 8))
    plt.ylim(-5, 105)
    plt.xlabel('t+nᵗʰ timestamp')
    plt.ylabel('Percentage')
    plt.grid(False)
    plt.legend()

    plt.tight_layout()

    plt.savefig('{}/{}.png'.format(REPORT_OUTPUT_DIR_PATH, dataset_name))


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (6, 3)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['legend.title_fontsize'] = 14
    plt.rcParams.update({'font.size': 14})

    # Create report dir
    if not os.path.exists(REPORT_OUTPUT_DIR_PATH):
        os.makedirs(REPORT_OUTPUT_DIR_PATH)

    for dataset_name, dataset_pretty_name in DATASET_NAME_LIST:
        result_output_dir_path = RESULT_OUTPUT_DIR_PATH_FORMAT.format(dataset_name)
        chosen_model_list = get_best_models(result_output_dir_path)
        plot_stuff(chosen_model_list, result_output_dir_path, dataset_name, dataset_pretty_name)
