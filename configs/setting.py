from pathlib import Path
import json


def setting(config_file: json = None, project: str = 'DAE'):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR.joinpath(f'session_{project}').mkdir(exist_ok=True)
    BASE_DIR.joinpath('trained_ae').mkdir(exist_ok=True)

    if config_file is None:
        config_name = 'CONFIG'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config['SEED'] = config_file['seed']

    config['DATASET_NAME'] = config_file['dataset']['name']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', config['DATASET_NAME'])
    config['NUM_WORKER'] = config_file['dataset']['n_worker']
    config['DEEPINSIGHT'] = config_file['dataset']['deepinsight']
    dataset_name = config['DATASET_NAME']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', dataset_name)

    config['AE_USE'] = config_file['autoencoder']['use_autoencoder']
    config['AE_TRAINABLE'] = config_file['autoencoder']['trainable']
    config['AE_FINETUNE'] = config_file['autoencoder']['fine_tuning']
    config['AE_HIDDEN_SIZE'] = config_file['autoencoder']['hidden_size']
    config['AE_EPOCH'] = config_file['autoencoder']['pretrain_epochs']
    config['AE_BATCH_SIZE'] = config_file['autoencoder']['pretrain_batch_size']
    config['AE_LOSS'] = config_file['autoencoder']['loss_fn']
    config['AE_ACTIVATION'] = config_file['autoencoder']['activation']
    config['AE_OPTIMIZER'] = config_file['autoencoder']['optimizer']
    config['AE_SCHEDULE'] = config_file['autoencoder']['lr_schedule']
    config['AE_INITIAL_LR'] = config_file['autoencoder']['initial_lr']
    config['AE_FINAL_LR'] = config_file['autoencoder']['finial_lr']
    config['AE_DECAY'] = config_file['autoencoder']['decay']
    config['SAVE_RECON_DATASET'] = config_file['autoencoder']['save_recon_datasets']

    config['MODEL_NAME'] = config_file['classifier']['name']
    config['EPOCHS'] = config_file['classifier']['epoch']
    config['LOSS'] = config_file['classifier']['loss_fn']
    config['BATCH_SIZE'] = config_file['classifier']['batch_size']
    config['OPTIMIZER'] = config_file['classifier']['optimizer']
    config['LR'] = config_file['classifier']['lr']
    config['DECAY'] = config_file['classifier']['decay']
    config['SCHEDULER'] = config_file['classifier']['scheduler']
    config['MIN_LR'] = config_file['classifier']['min_lr']
    config['MONITOR'] = config_file['classifier']['monitor']
    config['EARLY_STOP'] = config_file['classifier']['early_stop']
    config['REGULARIZATION'] = config_file['classifier']['regularization']

    return config, config_file
