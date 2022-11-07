from darts.models import AutoARIMA

from experiments.helpers_multi import run

DATASET_LIST = [
    ('boralasgamuwa_uc_2012-2018', '2016-05-01 00:00:00'),
    ('moratuwa_mc_2014-2018', '2017-11-01 00:00:00'),
    ('dehiwala_mc_2012-2018', '2015-02-01 00:00:00'),
    ('open_source_ballarat_daily_waste_2000_jul_2015_mar', '2008-09-01 00:00:00'),
    ('open_source_austin_daily_waste_2003_jan_2021_jul', '2015-06-01 00:00:00')
]


def generate_model_name(params):
    model_name = 'daily_arima_multi_weekdays-{}_diff-{}'.format(
        params['only_weekdays'],
        params['is_differenced'],
    )

    return model_name


def generate_model(params):
    model = AutoARIMA()

    return model


def run_tests():
    for dataset_name, test_split_before in DATASET_LIST:
        params = {
            'dataset_name': dataset_name,
            'test_split_before': test_split_before,
            'only_weekdays': False,
            'is_differenced': False,
        }
        run(params, generate_model_name, generate_model)
