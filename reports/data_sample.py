import os

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries

DATASET_NAME_LIST = [
    'boralasgamuwa_uc_2012-2018',
    'dehiwala_mc_2012-2018',
    'moratuwa_mc_2014-2018',
    'open_source_austin_daily_waste_2003_jan_2021_jul',
    'open_source_ballarat_daily_waste_2000_jul_2015_mar'
]

REPORT_OUTPUT_DIR_PATH = '../../tmp/univariate_forecast_daily/reports/data_sample'

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['figure.figsize'] = (14, 8)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['legend.title_fontsize'] = 30
    plt.rcParams.update({'font.size': 30})

    # Create result dir
    if not os.path.exists(REPORT_OUTPUT_DIR_PATH):
        os.makedirs(REPORT_OUTPUT_DIR_PATH)

    for dataset_name in DATASET_NAME_LIST:
        daily_waste_data_file_path = f'../../tmp/univariate_forecast_daily/{dataset_name}/imputed_data.csv'

        df = pd.read_csv(daily_waste_data_file_path)[:180]

        # Convert ticket_date to UTC
        df['ticket_date'] = pd.to_datetime(df['ticket_date']).dt.tz_localize(None)

        # Convert to a Darts Timeseries
        series = TimeSeries.from_dataframe(df, time_col='ticket_date', value_cols='net_weight_kg')

        fig, ax = plt.subplots()

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        series.plot()

        plt.xlabel('Date')
        plt.ylabel('MSW Net weight (kg)')
        plt.legend('')

        fig.tight_layout()

        plt.savefig('{}/{}.eps'.format(REPORT_OUTPUT_DIR_PATH, dataset_name), format='eps')
