import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_NAME_LIST = [
    'boralasgamuwa_uc_2012-2018',
    'dehiwala_mc_2012-2018',
    'moratuwa_mc_2014-2018',
    'open_source_austin_daily_waste_2003_jan_2021_jul',
    'open_source_ballarat_daily_waste_2000_jul_2015_mar'
]

REPORT_OUTPUT_DIR_PATH = '../../tmp/univariate_forecast_daily/reports/daily_box_plots'

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['figure.figsize'] = (14, 10)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.rcParams['legend.fontsize'] = 17
    plt.rcParams['legend.title_fontsize'] = 17
    plt.rcParams.update({'font.size': 17})

    # Create report dir
    if not os.path.exists(REPORT_OUTPUT_DIR_PATH):
        os.makedirs(REPORT_OUTPUT_DIR_PATH)

    for dataset_name in DATASET_NAME_LIST:
        daily_waste_data_file_path = f'../../tmp/univariate_forecast_daily/{dataset_name}/imputed_data.csv'

        df = pd.read_csv(daily_waste_data_file_path)

        df['ticket_date'] = df['ticket_date'].astype('datetime64[ns]')

        df['day_of_week'] = df['ticket_date'].dt.dayofweek

        df['day_of_week'] = df['day_of_week'].replace({
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        })

        PROPS = {
            'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
            'medianprops': {'color': 'black'},
            'whiskerprops': {'color': 'black'},
            'capprops': {'color': 'black'}
        }

        fig, ax = plt.subplots()

        ax = sns.boxplot(
            x="day_of_week",
            y="net_weight_kg",
            data=df,
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            **PROPS
        ).set(
            xlabel='Net weight (kg)',
            ylabel='Day of the week'
        )

        plt.savefig('{}/{}.eps'.format(REPORT_OUTPUT_DIR_PATH, dataset_name), format='eps')
