from pathlib import Path
import json


def setting_DAE(config_file: json = None, project: str = 'DAE'):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR.joinpath(f'session_{project}').mkdir(exist_ok=True)
    BASE_DIR.joinpath('trained_ae').mkdir(exist_ok=True)

    if config_file is None:
        config_name = 'CONFIG_DAE'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config['SEED'] = config_file['seed']

    config['DATASET_NAME'] = config_file['dataset']['name']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', config['DATASET_NAME'])
    config['VAL_SIZE'] = config_file['dataset']['val_size']
    config['NUM_WORKER'] = config_file['dataset']['n_worker']

    config['AE_TRAINABLE'] = config_file['autoencoder']['trainable']
    config['AE_FINETUNE'] = config_file['autoencoder']['fine_tuning']
    config['AE_METHOD'] = config_file['autoencoder']['ae_method']
    config['AE_UNIT'] = config_file['autoencoder']['ae_unit']
    config['AE_EPOCH'] = config_file['autoencoder']['pretrain_epoch']
    config['AE_BATCH'] = config_file['autoencoder']['ae_batch']
    config['AE_LOSS'] = config_file['autoencoder']['loss_fn']
    config['AE_ACTIVATION'] = config_file['autoencoder']['activation']
    config['AE_O_ACTIVATION'] = config_file['autoencoder']['o_activation']
    config['AE_OPTIMIZER'] = config_file['autoencoder']['optimizer']
    config['AE_SCHEDULE'] = config_file['autoencoder']['lr_schedule']
    config['AE_INITIAL_LR'] = config_file['autoencoder']['initial_lr']
    config['AE_FINAL_LR'] = config_file['autoencoder']['finial_lr']
    config['AE_DECAY'] = config_file['autoencoder']['decay']

    config['MODEL_NAME'] = config_file['classifier']['name']
    config['UNIT'] = config_file['classifier']['unit']
    config['MIN_DROPOUT'] = config_file['classifier']['min_dropout']
    config['MAX_DROPOUT'] = config_file['classifier']['max_dropout']
    config['BATCH'] = config_file['classifier']['batch']
    config['MIN_LR'] = config_file['classifier']['min_lr']
    config['MAX_LR'] = config_file['classifier']['max_lr']
    config['EPOCH'] = config_file['classifier']['epoch']
    config['LOSS'] = config_file['classifier']['loss_fn']
    config['MONITOR'] = config_file['classifier']['monitor']
    config['EARLY_STOP'] = config_file['classifier']['early_stop']
    config['MAX_EVALS'] = config_file['classifier']['max_evals']

    return config, config_file
