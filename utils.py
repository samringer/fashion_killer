import pickle
import random
from os import mkdir
from os.path import join, exists

import numpy as np
from absl import flags
import torch
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS

flags.DEFINE_string('task_path', None, "Path to save directory of task.")
flags.DEFINE_string('exp_name', None, "Name of the experiment being run")
flags.DEFINE_string('data_dir', None, "Path to dataset used in training")
flags.DEFINE_string('load_checkpoint', None,
                    "The path to load a checkpoint from")
# TODO: Infer this from torch.cuda.is_available
flags.DEFINE_boolean('use_cuda', True, "Whether to use GPU")
flags.DEFINE_boolean('use_fp16', True, "Whether to use mixed precision")
flags.DEFINE_boolean('over_train', False, "Overtrain on one datapoint")
flags.DEFINE_integer('checkpoint_interval', 5000,
                     "How often in train steps to checkpoint.")
flags.DEFINE_integer('tb_log_interval', 30,
                     "Training step interval for tensorboard logging.")

# Training params
flags.DEFINE_float('learning_rate', 1e-4, "Starting learning rate")
flags.DEFINE_integer('batch_size', 4, "Batch size to use when training")
flags.DEFINE_integer('num_epochs', 10, "Number of training epochs")

def save_checkpoint(model, optimizer, save_path):
    """
    Checkpoint both the model and optimizer.
    Save them both together in a tuple.
    """
    with open(save_path, 'wb') as out_f:
        pickle.dump((model.state_dict(), optimizer.state_dict()), out_f)


def load_checkpoint(model, optimizer):
    """
    Load the model and optimizer state dict from the path provided
    by FLAGS.load_checkpoint.
    """
    with open(FLAGS.load_checkpoint, 'rb') as in_f:
        model_state_dict, optimizer_state_dict = pickle.load(in_f)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    return model, optimizer


def prepare_experiment_dirs():
    """
    Create the experiment directory if it does not already exists.
    Also create subdirectories to store saved models and logs.
    """
    if not FLAGS.task_path:
        raise RuntimeError("all training runs must define task_path")
    if not FLAGS.exp_name:
        raise RuntimeError("all training runs must define exp_name")

    exp_root_path = join(FLAGS.task_path, FLAGS.exp_name)
    if not exists(exp_root_path):
        mkdir(exp_root_path)

    models_path = join(exp_root_path, 'models')
    if not exists(models_path):
        mkdir(models_path)

    logs_path = join(exp_root_path, 'logs')
    if not exists(logs_path):
        mkdir(logs_path)

    return models_path


def get_tb_logger():
    """
    Return the logger used for logging to tensorboard.
    Ensures the logging directory exists.
    """
    logs_path = join(FLAGS.task_path, FLAGS.exp_name, 'logs')
    assert exists(logs_path)
    return SummaryWriter(logs_path)


def set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)