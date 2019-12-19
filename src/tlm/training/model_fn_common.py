import tensorflow as tf

from tf_util.tf_logging import tf_logging
from trainer.get_param_num import get_param_num


def get_tpu_scaffold_or_init(init_fn, use_tpu):
    if use_tpu:
        def tpu_scaffold():
            init_fn()
            return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
        return scaffold_fn
    else:
        init_fn()
        return None


def log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2=None):
    tf_logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        if initialized_variable_names2 is not None:
            if var.name in initialized_variable_names2:
                init_string = ", *INIT_FROM_CKPT2*"
        tf_logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                     init_string)
    tf_logging.info("Total parameters : %d" % get_param_num())