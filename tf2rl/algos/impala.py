import time
import logging
import multiprocessing

from cpprb import ReplayBuffer, MPPrioritizedReplayBuffer


def import_tf():
    import tensorflow as tf
    if tf.config.experimental.list_physical_devices('GPU'):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    return tf



def explorer():
    pass

def learner():
    pass


def run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn):
    initialize_logger(logging_level=logging.getLevelName(args.logging_level))
