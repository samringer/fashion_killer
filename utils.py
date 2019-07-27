import pickle
import random
from os import mkdir
from os.path import join, exists

import numpy as np
from absl import flags, logging
import torch
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
flags.DEFINE_integer('num_epochs', 500, "Number of training epochs")

flags.mark_flag_as_required('task_path')
flags.mark_flag_as_required('exp_name')
flags.mark_flag_as_required('data_dir')


def save_checkpoint(model, optimizer, scheduler, step_num):
    """
    Checkpoint the model, optimizer and scheduler.
    Saves them all together in a tuple representing the train state.
    """
    exp_dir = join(FLAGS.task_path, FLAGS.exp_name)
    save_path = join(exp_dir, 'models', '{}.chk'.format(step_num))

    checkpoint_state = (model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        step_num)
    with open(save_path, 'wb') as out_f:
        pickle.dump(checkpoint_state, out_f)

    logging.info('Checkpointed at {}'.format(save_path))


def load_checkpoint(model, optimizer, scheduler):
    """
    Load the model, optimizer and scheduler state dicts from
    the path provided by FLAGS.load_checkpoint.
    """
    with open(FLAGS.load_checkpoint, 'rb') as in_f:
        checkpoint_state = pickle.load(in_f)
    model_sd, optimizer_sd, scheduler_sd, step_num = checkpoint_state
    model.load_state_dict(model_sd)
    optimizer.load_state_dict(optimizer_sd)
    scheduler.load_state_dict(scheduler_sd)
    logging.info('Loaded from checkpoint {}'.format(FLAGS.load_checkpoint))
    return model, optimizer, scheduler, step_num


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


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_model(model_path):
    try:
        model = torch.load(model_path)
    except RuntimeError:
        model = torch.load(model_path, map_location="cpu")
    return model
