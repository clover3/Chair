
from tlm.tf_logging import tf_logging, logging

tf_logging.info("This is TF log")

import tensorflow as tf
from absl import logging as ab_logging
from tlm.training.train_flags import *
import tlm.model.base as modeling
from tlm.training.input_fn import input_fn_builder_unmasked as input_fn_builder
from tlm.training.model_fn import model_fn_builder
from tlm.model.base import BertModel
import os
def main(_):
    tf_logging2 = logging.getLogger('tensorflow')
    tf_logging.setLevel(logging.INFO)
    ab_logging.info("This is ab logging")
    tf_logging.info("This is TF log")
    tf_logging2.info("TFLog 2")




if __name__ == "__main__":
    tf.compat.v1.app.run()
