import json
import os

import pandas as pd
from darts import TimeSeries
from darts.metrics import mae
from darts.metrics import mape
from darts.metrics import r2_score
from darts.metrics import rmse
from tqdm.auto import tqdm


def calculate_modified_metrics(dataset_name):
    result_output_dir_path = '../../tmp/univariate_forecast_daily/{}'.format(dataset_name)

    unmodified_dir_list = []
    for dir_name in os.listdir(result_output_dir_path):
        dir_path = os.path.abspath(dir_name)

        output_csv_exists = os.path.exists('{}/{}/output.csv'.format(result_output_dir_path, dir_name))
        summary_json_exists = os.path.exists('{}/{}/summary.json'.format(result_output_dir_path, dir_name))
        modified_output_csv_not_exists = not os.path.exists('{}/{}/modified_output.csv'.format(result_output_dir_path, dir_name))
        modified_summary_json_not_exists = not os.path.exists('{}/{}/modified_summary.json'.format(result_output_dir_path, dir_name))

        if output_csv_exists and summary_json_exists and (modified_output_csv_not_exists or modified_summary_json_not_exists):
            unmodified_dir_list.append(dir_name)

    for dir_name in tqdm(unmodified_dir_list):
        with open('{}/{}/summary.json'.format(result_output_dir_path, dir_name)) as f:
            summary = json.load(f)

        output_df = pd.read_csv('{}/{}/output.csv'.format(result_output_dir_path, dir_name))

        # Remove actual values that are zero
        output_df = output_df[output_df['actual_net_weight_kg'] != 0].reset_index()

        # Set negative predictions to zero
        output_df['predicted_net_weight_kg'] = output_df['predicted_net_weight_kg'].apply(lambda x : x if x > 0 else 0)

        output_actual_series = TimeSeries.from_dataframe(output_df[['actual_net_weight_kg']], value_cols='actual_net_weight_kg')
        output_predicted_series = TimeSeries.from_dataframe(output_df[['predicted_net_weight_kg']], value_cols='predicted_net_weight_kg')

        # MAPE over time
        mape_hist = []
        for i in range(len(output_df['actual_net_weight_kg'])):
            mape_hist_val = mape(output_actual_series[i: i + 1], output_predicted_series[i: i + 1])
            mape_hist.append(mape_hist_val)
        output_df['mape'] = mape_hist

        output_df.to_csv('{}/{}/modified_output.csv'.format(result_output_dir_path, dir_name), index=False)

        rmse_val = rmse(output_actual_series, output_predicted_series)
        mae_val = mae(output_actual_series, output_predicted_series)
        mape_val = mape(output_actual_series, output_predicted_series)
        r2_score_val = r2_score(output_actual_series, output_predicted_series)

        summary['rmse'] = rmse_val
        summary['mae'] = mae_val
        summary['mape'] = mape_val
        summary['r2_score'] = r2_score_val

        with open('{}/{}/modified_summary.json'.format(result_output_dir_path, dir_name), 'w') as f:
            json.dump(summary, f)
