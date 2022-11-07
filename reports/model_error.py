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
    latex_formatted_error_dict = {}

    for dataset_name in DATASET_NAME_LIST:
        training_output_dir_path = f'../../tmp/univariate_forecast_daily/{dataset_name}'

        calculate_modified_metrics(dataset_name)
        chosen_model_list = get_best_models(training_output_dir_path)

        model_detail_list = []
        for dir_name, model_name in chosen_model_list:
            with open('{}/{}/modified_summary.json'.format(training_output_dir_path, dir_name)) as json_file:
                summary = json.load(json_file)

            model_detail_list.append([
                summary['rmse'],
                summary['mae'],
                summary['mape'],
                model_name
            ])

        # Sort by rmse_val
        model_detail_list = sorted(model_detail_list, key=lambda x: x[0])

        # Print details
        for rmse_val, mae_val, mape_val, model_name, in model_detail_list:
            if model_name not in latex_formatted_error_dict:
                latex_formatted_error_dict[model_name] = []
            latex_formatted_error_dict[model_name].append(
                '& {:.2f} & {:.2f} & {:.2f}\%'.format(rmse_val, mae_val, mape_val))

    print('################ Boralesgamuwa | Moratuwa | Dehiwala')

    print('\\multirow{9}{*}{\\rotatebox[origin=c]{90}{Single-model}}')
    for model_name in SINGLE_MODEL_NAME_LIST:
        print('\t& {}'.format(model_name[9:]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][0]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][1]))
        print('\t{} \\\\'.format(latex_formatted_error_dict[model_name][2]))
        print('')

    print('\hline')

    print('\\multirow{8}{*}{\\rotatebox[origin=c]{90}{Multi-model}}')
    for model_name in MULTI_MODEL_NAME_LIST:
        print('\t& {}'.format(model_name[8:]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][0]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][1]))
        print('\t{} \\\\'.format(latex_formatted_error_dict[model_name][2]))
        print('')

    print('################ Ballarat | Austin')

    print('\\multirow{9}{*}{\\rotatebox[origin=c]{90}{Single-model}}')
    for model_name in SINGLE_MODEL_NAME_LIST:
        print('\t& {}'.format(model_name[9:]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][3]))
        print('\t{} \\\\'.format(latex_formatted_error_dict[model_name][4]))
        print('')

    print('\hline')

    print('\\multirow{8}{*}{\\rotatebox[origin=c]{90}{Multi-model}}')
    for model_name in MULTI_MODEL_NAME_LIST:
        print('\t& {}'.format(model_name[8:]))
        print('\t{}'.format(latex_formatted_error_dict[model_name][3]))
        print('\t{} \\\\'.format(latex_formatted_error_dict[model_name][4]))
        print('')
