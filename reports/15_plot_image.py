import json
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (14, 20)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['legend.title_fontsize'] = 17
plt.rcParams.update({'font.size': 17})

N_DAYS = 30

RESULT_OUTPUT_DIR_PATH_FORMAT = '../../tmp/univariate_forecast_daily/{}'

DATASET_NAME_LIST = [
    ['boralasgamuwa_uc_2012-2018', 'Boralesgamuwa'],
    ['dehiwala_mc_2012-2018', 'Dehiwala'],
    ['moratuwa_mc_2014-2018', 'Moratuwa'],
    ['open_source_austin_daily_waste_2003_jan_2021_jul', 'Austin'],
    ['open_source_ballarat_daily_waste_2000_jul_2015_mar', 'Ballarat']
]

ML_MODEL_LIST = [
    ['daily_lr_weekdays', '[Single] Linear Regression'],
    ['daily_arima_weekdays', '[Single] Auto ARIMA'],
    ['daily_rf_weekdays', '[Single] Random Forest'],
    ['daily_light_gbm_weekdays', '[Single] Light GBM'],
    ['daily_prophet_weekdays', '[Single] Prophet'],

    # MULTI ##############################################################
    ['daily_lr_multi_weekdays', '[Ensemble] Linear Regression'],
    ['daily_arima_multi_weekdays', '[Ensemble] Auto ARIMA'],
    ['daily_rf_multi_weekdays', '[Ensemble] Random Forest'],
    ['daily_light_gbm_multi_weekdays', '[Ensemble] Light GBM'],
]

DL_MODEL_LIST = [
    ['daily_lstm_weekdays', '[Single] LSTM'],
    ['daily_tcn_weekdays', '[Single] TCN'],
    ['daily_transformer_weekdays', '[Single] Transformer'],
    ['daily_n_beats_weekdays', '[Single] N-BEATS'],

    # MULTI ##############################################################
    ['daily_lstm_multi_weekdays', '[Ensemble] LSTM'],
    ['daily_tcn_multi_weekdays', '[Ensemble] TCN'],
    ['daily_transformer_multi_weekdays', '[Ensemble] Transformer'],
    ['daily_n_beats_multi_weekdays', '[Ensemble] N-BEATS']
]


def get_best_models(result_output_dir_path, model_list):
    mape_list = []
    for prefix, model_name in model_list:
        for dir_name in os.listdir(result_output_dir_path):
            dir_path = os.path.abspath(dir_name)

            try:
                if dir_name.startswith(prefix):
                    with open(f'{result_output_dir_path}/{dir_name}/modified_summary.json') as f:
                        summary = json.load(f)

                    mape_list.append([summary['mape'], dir_name, model_name])  # Choose by validation score
            except:
                # print('Error: {}'.format(dir_name))
                pass

    # Sort by val MAPE
    mape_list = sorted(mape_list, key=lambda x: x[0])

    # => [dir_name, model_name]
    return (mape_list[0][1], mape_list[0][2])


def plot_actual_vs_predicted(result_output_dir_path, dir_name, model_name, ax):
    test_df = pd.read_csv(f'{result_output_dir_path}/{dir_name}/modified_output.csv')

    test_df['actual_net_weight_kg'][:N_DAYS].plot(
        ax=ax,
        label='Actual series',
        color='#388E3C',
        marker='D',
        markerfacecolor='none'
    )
    test_df['predicted_net_weight_kg'][:N_DAYS].plot(
        ax=ax,
        label='{}'.format(model_name),
        color='#1976D2',
        marker='s',
        markerfacecolor='none'
    )

    ax.set_xticks([0, 10, 20, 30], [1, 10, 20, 30])
    ax.set_xlabel('t+nᵗʰ timestamp')
    ax.set_ylabel('RMSE')

    ax.grid(False)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(-0.05, 1.3))


def plot_mape(result_output_dir_path, ml_dir_name, ml_model_name, dl_dir_name, dl_model_name, ax):
    ml_test_df = pd.read_csv(f'{result_output_dir_path}/{ml_dir_name}/modified_output.csv')
    dl_test_df = pd.read_csv(f'{result_output_dir_path}/{dl_dir_name}/modified_output.csv')

    ml_test_df['mape'][:N_DAYS].plot(
        ax=ax,
        label='Best ML model',
        color='#E64A19',
        marker='o',
        markerfacecolor='none'
    )
    dl_test_df['mape'][:N_DAYS].plot(
        ax=ax,
        label='Best DL model',
        color='#512DA8',
        marker='x',
        markerfacecolor='none'
    )

    ax.set_xticks([0, 10, 20, 30], [1, 10, 20, 30])
    ax.set_xlabel('t+nᵗʰ timestamp')
    ax.set_ylabel('Percentage')

    ax.grid(False)

    ax.set_ylim(-5, 105)
    ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(-0.05, 1.3))


fig, axes = plt.subplots(nrows=len(DATASET_NAME_LIST), ncols=3)

cols = [
    'Best ML model',
    'Best DL model',
    'MAPE'
]

for ax, col in zip(axes[0], cols):
    ax.set_title(
        col,
        fontweight='bold',
        color='black',
        x=0.5,
        y=1.35
    )

for ax, ds_name in zip(axes[:, 0], DATASET_NAME_LIST):
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.set_ylabel(
        ds_name[1],
        rotation=90,
        size=20,
        fontweight='bold',
        color='black',
        labelpad=-350
    )

for n_row in range(len(DATASET_NAME_LIST)):
    dataset_name, dataset_pretty_name = DATASET_NAME_LIST[n_row]
    result_output_dir_path = RESULT_OUTPUT_DIR_PATH_FORMAT.format(dataset_name)

    for n_col in range(3):
        ax = axes[n_row, n_col]

        if n_col == 0:
            best_ml_model = get_best_models(result_output_dir_path, ML_MODEL_LIST)

            plot_actual_vs_predicted(
                result_output_dir_path,
                best_ml_model[0],
                best_ml_model[1],
                ax
            )
        elif n_col == 1:
            best_dl_model = get_best_models(result_output_dir_path, DL_MODEL_LIST)

            plot_actual_vs_predicted(
                result_output_dir_path,
                best_dl_model[0],
                best_dl_model[1],
                ax
            )
        else:
            best_ml_model = get_best_models(result_output_dir_path, ML_MODEL_LIST)
            best_dl_model = get_best_models(result_output_dir_path, DL_MODEL_LIST)

            plot_mape(
                result_output_dir_path,
                best_ml_model[0],
                best_ml_model[1],
                best_dl_model[0],
                best_dl_model[1],
                ax
            )

fig.tight_layout()

plt.savefig('../../tmp/univariate_forecast_daily/reports/15_plot_image.png')
