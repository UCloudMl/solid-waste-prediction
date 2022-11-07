import json
import os

MODEL_LIST = [
    ['[Single] Linear Regression', 'daily_lr_weekdays'],
    ['[Single] Auto ARIMA', 'daily_arima_weekdays'],
    ['[Single] Random Forest', 'daily_rf_weekdays'],
    ['[Single] Light GBM', 'daily_light_gbm_weekdays'],
    ['[Single] Prophet', 'daily_prophet_weekdays'],
    ['[Single] LSTM', 'daily_lstm_weekdays'],
    ['[Single] TCN', 'daily_tcn_weekdays'],
    ['[Single] Transformer', 'daily_transformer_weekdays'],
    ['[Single] N-BEATS', 'daily_n_beats_weekdays'],

    # MULTI ##############################################################
    ['[Multi] Linear Regression', 'daily_lr_multi_weekdays'],
    ['[Multi] Auto ARIMA', 'daily_arima_multi_weekdays'],
    ['[Multi] Random Forest', 'daily_rf_multi_weekdays'],
    ['[Multi] Light GBM', 'daily_light_gbm_multi_weekdays'],
    ['[Multi] Prophet', 'daily_prophet_weekdays'],
    ['[Multi] LSTM', 'daily_lstm_multi_weekdays'],
    ['[Multi] TCN', 'daily_tcn_multi_weekdays'],
    ['[Multi] Transformer', 'daily_transformer_multi_weekdays'],
    ['[Multi] N-BEATS', 'daily_n_beats_multi_weekdays']
]


def get_best_models(result_output_dir_path):
    chosen_model_list = []
    for model_name, prefix in MODEL_LIST:
        rmse_list = []
        for dir_name in os.listdir(result_output_dir_path):
            dir_path = os.path.abspath(dir_name)

            if dir_name.startswith(prefix):
                try:
                    with open('{}/{}/modified_summary.json'.format(result_output_dir_path, dir_name)) as f:
                        summary = json.load(f)
                        rmse_list.append([summary['mape'], dir_name])
                except:
                    # print('Error: {}'.format(dir_name))
                    pass

        # Sort by RMSE
        rmse_list = sorted(rmse_list, key=lambda x: x[0])

        try:
            # chosen_model_list => [dir_name, model_name]
            chosen_model_list.append([rmse_list[0][1], model_name])
        except:
            pass

    return chosen_model_list
