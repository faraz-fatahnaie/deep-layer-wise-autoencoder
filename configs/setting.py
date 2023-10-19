from pathlib import Path
import json


def setting(config_file=None):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR.joinpath('session').mkdir(exist_ok=True)

    if config_file is None:
        config_name = 'CONFIG'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config['DATASET_NAME'] = config_file['dataset']['name']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', config['DATASET_NAME'])
    config['NUM_WORKER'] = config_file['dataset']['n_worker']

    config['AE_EPOCH'] = config_file['autoencoder']['pretrain_epochs']
    config['AE_BATCH_SIZE'] = config_file['autoencoder']['pretrain_batch_size']
    config['AE_LOSS'] = config_file['autoencoder']['loss_fn']
    config['AE_ACTIVATION'] = config_file['autoencoder']['activation']
    config['AE_OPTIMIZER'] = config_file['autoencoder']['optimizer']
    config['AE_SCHEDULE'] = config_file['autoencoder']['lr_schedule']
    config['AE_INITIAL_LR'] = config_file['autoencoder']['initial_lr']
    config['AE_FINAL_LR'] = config_file['autoencoder']['finial_lr']
    config['AE_DECAY'] = config_file['autoencoder']['decay']

    config['MODEL_NAME'] = config_file['classifier']['name']
    config['EPOCHS'] = config_file['classifier']['epoch']
    config['LOSS'] = config_file['classifier']['loss_fn']
    config['BATCH_SIZE'] = config_file['classifier']['batch_size']
    config['OPTIMIZER'] = config_file['classifier']['optimizer']
    config['LR'] = config_file['classifier']['lr']
    config['DECAY'] = config_file['classifier']['decay']
    config['SCHEDULER'] = config_file['classifier']['scheduler']
    config['MIN_LR'] = config_file['classifier']['min_lr']
    config['REGULARIZATION'] = config_file['classifier']['regularization']

    return config, config_file