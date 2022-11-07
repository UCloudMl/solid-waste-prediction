from darts.models import TCNModel
from tqdm.auto import tqdm

from experiments.helpers_multi import run
from util.custom_progress_bar import CustomTQDMProgressBar

DATASET_LIST = [
    ('boralasgamuwa_uc_2012-2018', '2016-05-01 00:00:00'),
    ('moratuwa_mc_2014-2018', '2017-11-01 00:00:00'),
    ('dehiwala_mc_2012-2018', '2015-02-01 00:00:00'),
    ('open_source_ballarat_daily_waste_2000_jul_2015_mar', '2008-09-01 00:00:00'),
    ('open_source_austin_daily_waste_2003_jan_2021_jul', '2015-06-01 00:00:00')
]


def generate_model_name(params):
    model_name = 'daily_tcn_multi_weekdays-{}_diff-{}_inp-{}_out-{}_eph-{}_dil-{}_ken-{}_nfil-{}_drp-{}_bat-{}_wnrm-{}_lr-{}'.format(
        params['only_weekdays'],
        params['is_differenced'],
        params['input_chunk_length'],
        params['output_chunk_length'],
        params['n_epochs'],
        params['dilation_base'],
        params['kernel_size'],
        params['num_filters'],
        params['dropout'],
        params['batch_size'],
        params['weight_norm'],
        params['optimizer_kwargs']['lr']
    )

    return model_name


def generate_model(params):
    model = TCNModel(
        input_chunk_length=params['input_chunk_length'],
        output_chunk_length=params['output_chunk_length'],
        n_epochs=params['n_epochs'],
        dilation_base=params['dilation_base'],
        kernel_size=params['kernel_size'],
        num_filters=params['num_filters'],
        dropout=params['dropout'],
        # batch_size=params['batch_size'],
        optimizer_kwargs=params['optimizer_kwargs'],
        # weight_norm=params['weight_norm'],
        random_state=0,
        model_name=generate_model_name(params),
        log_tensorboard=False,
        force_reset=True,
        pl_trainer_kwargs={
            'accelerator': 'gpu',
            'gpus': [0],
            'enable_progress_bar': False,
            'enable_model_summary': False,
            'callbacks': [
                CustomTQDMProgressBar()
            ]
        }
    )

    return model


def run_tests():
    input_chunk_length_list = [4, 5, 6, 7, 8, 12, 16, 24, 36, 48]
    n_iter = len(input_chunk_length_list)

    for dataset_name, test_split_before in DATASET_LIST:
        with tqdm(total=n_iter) as pbar:
            for input_chunk_length in input_chunk_length_list:
                kernel_size = input_chunk_length - 1
                num_filters = 10
                dilation_base = 10
                params = {
                    'dataset_name': dataset_name,
                    'test_split_before': test_split_before,
                    'only_weekdays': False,
                    'is_differenced': False,

                    'input_chunk_length': input_chunk_length,
                    'output_chunk_length': 1,
                    'n_epochs': 1000,
                    'dilation_base': dilation_base,
                    'kernel_size': kernel_size,
                    'num_filters': num_filters,
                    'dropout': 0.2,
                    'batch_size': 0,
                    'optimizer_kwargs': {'lr': 1e-4},
                    'weight_norm': True,
                }
                run(params, generate_model_name, generate_model)
                pbar.update(1)
