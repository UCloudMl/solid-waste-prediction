from darts.models import TransformerModel
from tqdm.auto import tqdm

from experiments.helpers import run
from util.custom_progress_bar import CustomTQDMProgressBar

DATASET_LIST = [
    ('boralasgamuwa_uc_2012-2018', '2016-05-01 00:00:00'),
    ('moratuwa_mc_2014-2018', '2017-11-01 00:00:00'),
    ('dehiwala_mc_2012-2018', '2015-02-01 00:00:00'),
    ('open_source_ballarat_daily_waste_2000_jul_2015_mar', '2008-09-01 00:00:00'),
    ('open_source_austin_daily_waste_2003_jan_2021_jul', '2015-06-01 00:00:00')
]


def generate_model_name(params):
    model_name = 'daily_transformer_weekdays-{}_diff-{}_inp-{}_out-{}_eph-{}_dmod-{}_nhead-{}_nenc-{}_ndec-{}_dimf-{}_drp-{}_bat-{}_lr-{}'.format(
        params['only_weekdays'],
        params['is_differenced'],
        params['input_chunk_length'],
        params['output_chunk_length'],
        params['n_epochs'],
        params['d_model'],
        params['nhead'],
        params['num_encoder_layers'],
        params['num_decoder_layers'],
        params['dim_feedforward'],
        params['dropout'],
        params['batch_size'],
        params['optimizer_kwargs']['lr']
    )

    return model_name


def generate_model(params):
    model = TransformerModel(
        input_chunk_length=params['input_chunk_length'],
        output_chunk_length=params['output_chunk_length'],
        n_epochs=params['n_epochs'],
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_encoder_layers'],
        num_decoder_layers=params['num_decoder_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout'],
        # batch_size=params['batch_size'],
        optimizer_kwargs=params['optimizer_kwargs'],
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
    input_chunk_length_list = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    n_iter = len(input_chunk_length_list)

    for dataset_name, test_split_before in DATASET_LIST:
        with tqdm(total=n_iter) as pbar:
            for input_chunk_length in input_chunk_length_list:
                params = {
                    'dataset_name': dataset_name,
                    'test_split_before': test_split_before,
                    'only_weekdays': False,
                    'is_differenced': False,

                    'input_chunk_length': input_chunk_length,
                    'output_chunk_length': 1,
                    'n_epochs': 200,
                    'd_model': 16,
                    'nhead': 8,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dim_feedforward': 128,
                    'dropout': 0.2,
                    'batch_size': 0,
                    'optimizer_kwargs': {'lr': 1e-4},
                }
                try:
                    run(params, generate_model_name, generate_model)
                except:
                    print('Error: {}'.format(generate_model_name(params)))
                pbar.update(1)

    for dataset_name, test_split_before in DATASET_LIST:
        with tqdm(total=n_iter) as pbar:
            for input_chunk_length in input_chunk_length_list:
                params = {
                    'dataset_name': dataset_name,
                    'test_split_before': test_split_before,
                    'only_weekdays': False,
                    'is_differenced': False,

                    'input_chunk_length': input_chunk_length,
                    'output_chunk_length': 1,
                    'n_epochs': 500,
                    'd_model': 16,
                    'nhead': 8,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dim_feedforward': 128,
                    'dropout': 0.2,
                    'batch_size': 0,
                    'optimizer_kwargs': {'lr': 1e-4},
                }
                try:
                    run(params, generate_model_name, generate_model)
                except:
                    print('Error: {}'.format(generate_model_name(params)))
                pbar.update(1)


if __name__ == '__main__':
    run_tests()
