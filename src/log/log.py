import logging

import os
from os.path import dirname

project_root = os.path.abspath(dirname(dirname(dirname(os.path.abspath(__file__)))))
data_path = os.path.join(project_root, 'data')
output_path = os.path.join(project_root, 'output')

def train_logger():
    log = logging.getLogger('Training')
    ch = logging.FileHandler(os.path.join(output_path, "train.log"))
    log.setLevel(logging.DEBUG)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    return log


def aux_logger():
    log = logging.getLogger('Aux1')
    ch = logging.FileHandler(os.path.join(output_path, "aux.log"))
    log.setLevel(logging.DEBUG)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    return log


def temp_logger():
    log = logging.getLogger('contradiction_logger')
    ch = logging.FileHandler(os.path.join(output_path, "contradict.log"))
    log.setLevel(logging.DEBUG)
    format_str = '%(levelname) - %(asctime)-15s %(message)s'
    ch.setFormatter(format_str)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    return log