import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop


def parse_data(df, dataset_name: str, classification_mode: str, mode: str = 'np'):
    classes = []
    if classification_mode == 'binary':
        classes = df.columns[-2:]
    elif classification_mode == 'multi':
        if dataset_name in ['NSL_KDD', 'KDD_CUP99']:
            classes = df.columns[-5:]
        elif dataset_name == 'UNSW_NB15':
            classes = df.columns[-10:]
        elif dataset_name == 'CICIDS':
            classes = df.columns[-15:]

    assert classes is not None, 'Something Wrong!!\nno class columns could be extracted from dataframe'
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(c) for c in list(classes)])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb


def shuffle_dataframe(dataframe_path):
    dataframe = pd.read_csv(dataframe_path)
    shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    base_path = os.path.dirname(dataframe_path)
    base_name = os.path.splitext(os.path.basename(dataframe_path))[0]
    file_extension = os.path.splitext(dataframe_path)[-1]
    new_file_name = Path(base_path).joinpath(base_name + file_extension)

    shuffled_dataframe.to_csv(new_file_name, index=False)

    print(f"Shuffled DataFrame saved as {new_file_name}")


class OptimizerFactory:
    def __init__(self,
                 opt: str = 'adam',
                 lr_schedule: bool = True,
                 len_dataset: int = 494021,
                 epochs: int = 50,
                 batch_size: int = 100,
                 init_lr: float = 0.1,
                 final_lr: float = 0.00001):
        self.opt = opt
        self.lr_schedule = lr_schedule
        self.len_dataset = len_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.final_lr = final_lr

    def lr_scheduler(self):
        pretraining_learning_rate_decay_factor = (self.final_lr / self.init_lr) ** (1 / self.epochs)
        pretraining_steps_per_epoch = int(self.len_dataset / self.batch_size)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=pretraining_steps_per_epoch,
            decay_rate=pretraining_learning_rate_decay_factor,
            staircase=True)
        return lr_scheduler

    def get_opt(self):
        if self.opt == 'adam':
            if self.lr_schedule:
                return Adam(self.lr_scheduler())
            else:
                return Adam(learning_rate=self.init_lr)

        elif self.opt == 'sgd':
            if self.lr_schedule:
                return SGD(self.lr_scheduler())
            else:
                return SGD(learning_rate=5, decay=0.5, momentum=.85, nesterov=True)
                # return SGD(learning_rate=.1, decay=0.001, momentum=.95, nesterov=True)

        elif self.opt == 'rmsprop':
            if self.lr_schedule:
                return RMSprop(self.lr_schedule)
            else:
                return RMSprop(learning_rate=self.init_lr)


if __name__ == "__main__":
    df_path = 'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\' \
              'KDD_CUP99\\train_binary_2neuron_labelOnehot.csv'
    shuffle_dataframe(df_path)
