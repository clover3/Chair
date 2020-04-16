import warnings

from tlm.model.freeze_bert import FreezeEmbedding
from trainer.estimator_main_v2 import main_inner

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from tlm.training.train_flags import *


def main(_):
    return main_inner(FreezeEmbedding)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
