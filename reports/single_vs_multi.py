import json
from statistics import mean

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

REPORT_OUTPUT_DIR_PATH_PATTERN = '../../tmp/univariate_forecast_daily/reports/average_bar_plot/{}'

if __name__ == '__main__':

    rmse_dict = {}
    mae_dict = {}
    mape_dict = {}
    training_time_dict = {}
    predicting_time_dict = {}

    model_detail_list = []
    for dataset_name in DATASET_NAME_LIST:
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

    for training_type in ['single', 'multi']:
        rmse_dict[training_type] = []
        mae_dict[training_type] = []
        mape_dict[training_type] = []
        training_time_dict[training_type] = []
        predicting_time_dict[training_type] = []

    for rmse_val, mae_val, mape_val, training_time, predicting_time, model_name in model_detail_list:
        if model_name in SINGLE_MODEL_NAME_LIST:
            training_type = 'single'
        elif model_name in MULTI_MODEL_NAME_LIST:
            training_type = 'multi'

        rmse_dict[training_type].append(rmse_val)
        mae_dict[training_type].append(mae_val)
        mape_dict[training_type].append(mape_val)
        training_time_dict[training_type].append(training_time)
        predicting_time_dict[training_type].append(predicting_time)

    for training_type in ['single', 'multi']:
        print('#### {}'.format(training_type))
        print('Avg. RMSE: {:.2f}'.format(mean(rmse_dict[training_type])))
        print('Avg. MAE: {:.2f}'.format(mean(mae_dict[training_type])))
        print('Avg. MAPE: {:.2f}'.format(mean(mape_dict[training_type])))
        print('Avg. training time: {:.2f}'.format(mean(training_time_dict[training_type])))
        print('Avg. predicting time: {:.2f}'.format(mean(predicting_time_dict[training_type])))
