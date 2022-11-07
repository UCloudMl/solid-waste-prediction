import json
from statistics import mean

from univariate_forecast_daily.util.metrics import calculate_modified_metrics
from univariate_forecast_daily.util.model import get_best_models

AVERAGE_DATASET_SET_LIST = [
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
        'lk',
        (
            'boralasgamuwa_uc_2012-2018',
            'moratuwa_mc_2014-2018',
            'dehiwala_mc_2012-2018',
        )
    ],
    [
        'all',
        (
            'boralasgamuwa_uc_2012-2018',
            'moratuwa_mc_2014-2018',
            'dehiwala_mc_2012-2018',
            'open_source_ballarat_daily_waste_2000_jul_2015_mar',
            'open_source_austin_daily_waste_2003_jan_2021_jul',
        )
    ]
]

SINGLE_ML_MODEL_NAME_LIST = [
    '[Single] Linear Regression',
    '[Single] Auto ARIMA',
    '[Single] Random Forest',
    '[Single] Light GBM',
    '[Single] Prophet',
]

MULTI_ML_MODEL_NAME_LIST = [
    '[Multi] Linear Regression',
    '[Multi] Auto ARIMA',
    '[Multi] Random Forest',
    '[Multi] Light GBM',
    '[Multi] Prophet',
]

SINGLE_DL_MODEL_NAME_LIST = [
    '[Single] LSTM',
    '[Single] TCN',
    '[Single] Transformer',
    '[Single] N-BEATS'
]

MULTI_DL_MODEL_NAME_LIST = [
    '[Multi] LSTM',
    '[Multi] TCN',
    '[Multi] Transformer',
    '[Multi] N-BEATS'
]

if __name__ == '__main__':
    for average_dataset_set_name, average_dataset_set in AVERAGE_DATASET_SET_LIST:
        print('------------------------------------------------------------------------')
        print('#### {}'.format(average_dataset_set_name))

        rmse_dict = {}
        mae_dict = {}
        mape_dict = {}
        training_time_dict = {}
        predicting_time_dict = {}
        for dataset_name in average_dataset_set:
            training_output_dir_path = f'../../tmp/univariate_forecast_daily/{dataset_name}'

            calculate_modified_metrics(dataset_name)
            chosen_model_list = get_best_models(training_output_dir_path)

            for dir_name, model_name in chosen_model_list:
                with open(f'{training_output_dir_path}/{dir_name}/modified_summary.json') as json_file:
                    summary = json.load(json_file)

                if model_name not in rmse_dict:
                    rmse_dict[model_name] = []
                    mae_dict[model_name] = []
                    mape_dict[model_name] = []
                    training_time_dict[model_name] = []
                    predicting_time_dict[model_name] = []

                rmse_dict[model_name].append(summary['rmse'])
                mae_dict[model_name].append(summary['mae'])
                mape_dict[model_name].append(summary['mape'])
                training_time_dict[model_name].append(summary['training_time'])
                predicting_time_dict[model_name].append(summary['predicting_time'])

        output_combinations = [
            # ['single-ml', SINGLE_ML_MODEL_NAME_LIST],
            # ['multi-ml', MULTI_ML_MODEL_NAME_LIST],
            # ['single-dl', SINGLE_DL_MODEL_NAME_LIST],
            # ['multi-dl', MULTI_DL_MODEL_NAME_LIST],
            ['single', SINGLE_ML_MODEL_NAME_LIST + SINGLE_DL_MODEL_NAME_LIST],
            ['multi', MULTI_ML_MODEL_NAME_LIST + MULTI_DL_MODEL_NAME_LIST],
            ['ml', SINGLE_ML_MODEL_NAME_LIST + MULTI_ML_MODEL_NAME_LIST],
            ['dl', SINGLE_DL_MODEL_NAME_LIST + MULTI_DL_MODEL_NAME_LIST],
            ['all', SINGLE_ML_MODEL_NAME_LIST + MULTI_ML_MODEL_NAME_LIST + SINGLE_DL_MODEL_NAME_LIST + MULTI_DL_MODEL_NAME_LIST],
        ]

        for comb_name, comb_models in output_combinations:
            print(f'{comb_name}')
            print(f'Avg. RMSE: {mean([v_child for k, v in rmse_dict.items() for v_child in v if k in comb_models]):.2f}')
            print(f'Avg. MAE: {mean([v_child for k, v in mae_dict.items() for v_child in v if k in comb_models]):.2f}')
            print(f'Avg. MAPE: {mean([v_child for k, v in mape_dict.items() for v_child in v if k in comb_models]):.2f}')
            print(f'Avg. training time: {mean([v_child for k, v in training_time_dict.items() for v_child in v if k in comb_models]):.2f}')
            print(f'Avg. predicting time: {mean([v_child for k, v in predicting_time_dict.items() for v_child in v if k in comb_models]):.2f}')
