from darts.models import LightGBMModel
from tqdm.auto import tqdm

from experiments.helpers_multi import run

DATASET_LIST = [
    ('boralasgamuwa_uc_2012-2018', '2016-05-01 00:00:00'),
    ('moratuwa_mc_2014-2018', '2017-11-01 00:00:00'),
    ('dehiwala_mc_2012-2018', '2015-02-01 00:00:00'),
    ('open_source_ballarat_daily_waste_2000_jul_2015_mar', '2008-09-01 00:00:00'),
    ('open_source_austin_daily_waste_2003_jan_2021_jul', '2015-06-01 00:00:00')
]


def generate_model_name(params):
    model_name = 'daily_light_gbm_multi_weekdays-{}_diff-{}_lags-{}'.format(
        params['only_weekdays'],
        params['is_differenced'],
        params['n_lags']
    )

    return model_name


def generate_model(params):
    model = LightGBMModel(
        lags=params['n_lags'],
    )

    return model


def run_tests():
    n_lags_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 36, 48]
    n_iter = len(n_lags_list)

    for dataset_name, test_split_before in DATASET_LIST:
        with tqdm(total=n_iter) as pbar:
            for n_lags in n_lags_list:
                try:
                    params ={
                        'dataset_name': dataset_name,
                        'test_split_before': test_split_before,
                        'only_weekdays': False,
                        'is_differenced': False,

                        'n_lags': n_lags
                    }
                    run(params, generate_model_name, generate_model)
                except:
                    print('[Failure] n_lags:{}'.format(n_lags))

                pbar.update(1)
