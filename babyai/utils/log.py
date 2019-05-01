# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import numpy
import logging
from collections import OrderedDict

from .. import utils


def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)


def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")


def synthesize(array):
    stats = OrderedDict()
    stats['mean'] = numpy.mean(array)
    stats['std'] = numpy.std(array)
    stats['min'] = numpy.min(array)
    stats['max'] = numpy.max(array)
    return stats
    '''return {
        "mean": numpy.mean(array),
        "std": numpy.std(array),
        "min": numpy.amin(array),
        "max": numpy.amax(array)
    }'''


def get_logger(log_name):
    path = get_log_path(log_name)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()
