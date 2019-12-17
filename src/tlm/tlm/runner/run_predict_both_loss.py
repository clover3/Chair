import warnings
from collections import Counter

from tf_util.tf_logging import tf_logging, logging
from tlm.tlm.model_fn_try_all_loss import model_fn_try_all_loss
from tlm.training import flags_wrapper
from tlm.training.dynamic_mask_main import LMTrainConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
import tlm.model.base as modeling
from tlm.training.input_fn import input_fn_builder_unmasked
from trainer.tpu_estimator import run_estimator, show_input_files
from tlm.training.train_flags import *


class CounterFilter(logging.Filter):
    targets = ["Dequeue next", "Enqueue next"]
    counter = Counter()
    def filter(self, record):
        for e in self.targets:
            if e in record.msg:
                self.counter[e] += 1
                record.msg += " ({})".format(self.counter[e])
                return True
        return True


def main(_):
    tf_logging.addFilter(CounterFilter())
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = flags_wrapper.get_input_files()
    train_config = LMTrainConfig.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = model_fn_try_all_loss(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
    )
    if FLAGS.do_predict:
        input_fn = input_fn_builder_unmasked(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)
    else:
        assert False

    r = run_estimator(model_fn, input_fn)
    return r


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
