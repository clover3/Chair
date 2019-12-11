

import tensorflow as tf

from tlm.training import train_flags


def main(_):
    flags = train_flags.FLAGS
    print(flags.max_def_length)
    print(flags.key_flags_by_module_dict())
    log_flags = ["max_seq_length"]
    for key in log_flags:
        value = getattr(flags, key)
        print("{}:\t{}\n".format(key, value))


if __name__ == "__main__":
    tf.compat.v1.app.run()
