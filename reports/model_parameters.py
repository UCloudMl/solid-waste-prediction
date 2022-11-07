import json

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
    latex_formatted_params_dict = {}

    for dataset_name in DATASET_NAME_LIST:
        training_output_dir_path = f'../../tmp/univariate_forecast_daily/{dataset_name}'

        calculate_modified_metrics(dataset_name)
        chosen_model_list = get_best_models(training_output_dir_path)

        model_detail_list = []
        for dir_name, model_name in chosen_model_list:
            with open(f'{training_output_dir_path}/{dir_name}/params.json') as json_file:
                params = json.load(json_file)

            model_detail_list.append([
                model_name,
                params
            ])

        # Sort by rmse_val
        model_detail_list = sorted(model_detail_list, key=lambda x: x[0])

        for model_name, params in model_detail_list:
            if model_name not in latex_formatted_params_dict:
                latex_formatted_params_dict[model_name] = []
            latex_formatted_params_dict[model_name].append(params)

    print('################ Single-model')

    for model_name in SINGLE_MODEL_NAME_LIST:
        param_name_list = list(latex_formatted_params_dict[model_name][0].keys())
        param_name_list.remove('dataset_name')
        param_name_list.remove('test_split_before')
        param_name_list.remove('only_weekdays')
        param_name_list.remove('is_differenced')
        if 'batch_size' in param_name_list:
            param_name_list.remove('batch_size')

        print('\\multirow{{{}}}{{*}}{{{}}}'.format(len(param_name_list), model_name[9:]))

        for param_name in param_name_list:
            param_value_string = ''
            for area_params in latex_formatted_params_dict[model_name]:
                param_value = area_params[param_name]
                param_value_string += ' & {}'.format(param_value)

            print('\t& {}{} \\\\'.format(param_name.replace('_', '\_'), param_value_string))

        print('\hline')
        print()

    print('################ Multi-model')

    for model_name in MULTI_MODEL_NAME_LIST:
        param_name_list = list(latex_formatted_params_dict[model_name][0].keys())
        param_name_list.remove('dataset_name')
        param_name_list.remove('test_split_before')
        param_name_list.remove('only_weekdays')
        param_name_list.remove('is_differenced')
        if 'batch_size' in param_name_list:
            param_name_list.remove('batch_size')

        print('\\multirow{{{}}}{{*}}{{{}}}'.format(len(param_name_list), model_name[8:]))

        for param_name in param_name_list:
            param_value_string = ''
            for area_params in latex_formatted_params_dict[model_name]:
                param_value = area_params[param_name]
                param_value_string += ' & {}'.format(param_value)

            print('\t& {}{} \\\\'.format(param_name.replace('_', '\_'), param_value_string))

        print('\hline')
        print()
