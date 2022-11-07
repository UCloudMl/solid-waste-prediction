import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['figure.figsize'] = (12, 20)
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['xtick.labelsize'] = 34
# plt.rcParams['ytick.labelsize'] = 34
# plt.rcParams['legend.fontsize'] = 34
# plt.rcParams['legend.title_fontsize'] = 34
# plt.rcParams.update({'font.size': 34})

font_size = 110

plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['legend.title_fontsize'] = font_size
plt.rcParams.update({'font.size': font_size})

N_DAYS = 60

RESULT_OUTPUT_DIR_PATH_FORMAT = '../../tmp/univariate_forecast_daily/{}'

DATASET_NAME_LIST = [
    # ['boralasgamuwa_uc_2012-2018', 'Boralesgamuwa'],
    # ['dehiwala_mc_2012-2018', 'Dehiwala'],
    ['moratuwa_mc_2014-2018', 'Moratuwa'],
    # ['open_source_austin_daily_waste_2003_jan_2021_jul', 'Austin'],
    ['open_source_ballarat_daily_waste_2000_jul_2015_mar', 'Ballarat']
]

ML_MODEL_LIST = [
    ['daily_lr_weekdays', '[Single] Linear\nRegression'],
    ['daily_arima_weekdays', '[Single] Auto\nARIMA'],
    ['daily_rf_weekdays', '[Single] Random\nForest'],
    ['daily_light_gbm_weekdays', '[Single] Light\nGBM'],
    ['daily_prophet_weekdays', '[Single] Prophet'],

    # MULTI ##############################################################
    ['daily_lr_multi_weekdays', '[Multi] Linear\nRegression'],
    ['daily_arima_multi_weekdays', '[Multi] Auto\nARIMA'],
    ['daily_rf_multi_weekdays', '[Multi] Random\nForest'],
    ['daily_light_gbm_multi_weekdays', '[Multi] Light\nGBM'],
]

DL_MODEL_LIST = [
    ['daily_lstm_weekdays', '[Single] LSTM'],
    ['daily_tcn_weekdays', '[Single] TCN'],
    ['daily_transformer_weekdays', '[Single] Transformer'],
    ['daily_n_beats_weekdays', '[Single] N-BEATS'],

    # MULTI ##############################################################
    ['daily_lstm_multi_weekdays', '[Multi] LSTM'],
    ['daily_tcn_multi_weekdays', '[Multi] TCN'],
    ['daily_transformer_multi_weekdays', '[Multi] Transformer'],
    ['daily_n_beats_multi_weekdays', '[Multi] N-BEATS']
]


def get_best_models(result_output_dir_path, model_list):
    rmse_list = []
    for prefix, model_name in model_list:
        for dir_name in os.listdir(result_output_dir_path):
            dir_path = os.path.abspath(dir_name)

            try:
                if dir_name.startswith(prefix):
                    with open('{}/{}/modified_summary.json'.format(result_output_dir_path, dir_name)) as f:
                        summary = json.load(f)

                    rmse_list.append([summary['mape'], dir_name, model_name])  # Choose by validation score
            except:
                # print('Error: {}'.format(dir_name))
                pass

    # Sort by val MAPE
    rmse_list = sorted(rmse_list, key=lambda x: x[0])

    # => [dir_name, model_name]
    return (rmse_list[0][1], rmse_list[0][2])


def plot_actual_vs_predicted(result_output_dir_path, dir_name, model_name, output_name, output_dir):
    test_df = pd.read_csv('{}/{}/modified_output.csv'.format(result_output_dir_path, dir_name))

    fig, ax = plt.subplots()

    test_df['actual_net_weight_kg'][:N_DAYS].plot(
        ax=ax,
        label='Actual series',
        color='#1A237E',
        marker='D',
        markersize=2,
        markerfacecolor='none'
    )
    test_df['predicted_net_weight_kg'][:N_DAYS].plot(
        ax=ax,
        label='{}'.format(model_name),
        color='#BF360C',
        marker='s',
        markersize=2,
        markerfacecolor='none'
    )

    # ax.set_xticks([0, 10, 20, 30], [1, 10, 20, 30])
    ax.set_xticks([0, 20, 40, 60], [1, 20, 40, 60])
    # ax.set_xticks([0, 30, 60, 90], [1, 30, 60, 90])
    ax.set_xlabel('t+nᵗʰ timestamp')
    ax.set_ylabel('Net weight (kg)')

    ax.grid(False)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(0.05, 1.02, 1, 0.2), ncol=2, frameon=False)
    lgd = ax.legend(loc='upper left', labelspacing=0.1, bbox_to_anchor=(-0.2, 1, 0.5, 1), frameon=False)
    # ax.legend(frameon=False)

    plt.gcf().subplots_adjust(left=0.2)

    for item in [fig, ax]:
        item.patch.set_visible(False)

    # fig.tight_layout()

    # plt.savefig('{}/{}.eps'.format(output_dir, output_name), format='eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('{}/{}.eps'.format(output_dir, output_name), format='eps', bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    for dataset_name, dataset_pretty_name in DATASET_NAME_LIST:
        result_output_dir_path = RESULT_OUTPUT_DIR_PATH_FORMAT.format(dataset_name)

        output_dir = '../../tmp/univariate_forecast_daily/reports/mini_plots'

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Best ML model plot
        best_ml_model_dir, best_ml_model_name = get_best_models(result_output_dir_path, ML_MODEL_LIST)
        output_name = '{}_{}'.format(dataset_name, 'best_ml')
        plot_actual_vs_predicted(
            result_output_dir_path,
            best_ml_model_dir,
            best_ml_model_name,
            output_name,
            output_dir
        )

        # Best DL model plot
        best_dl_model_dir, best_dl_model_name = get_best_models(result_output_dir_path, DL_MODEL_LIST)
        output_name = '{}_{}'.format(dataset_name, 'best_dl')
        plot_actual_vs_predicted(
            result_output_dir_path,
            best_dl_model_dir,
            best_dl_model_name,
            output_name,
            output_dir
        )

        # Combined MAPE plot
        # TODO
