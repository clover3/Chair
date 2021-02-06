
import tensorflow as tf

from tf_util.tf_logging import tf_logging, set_level_debug, check, reset_root_log_handler

check("point1")

def main(_):
    print("Main")
    check("point2")
    tf_logging.info("Log")

    reset_root_log_handler()
    set_level_debug()
    check("point3")
    tf_logging.info("Log")


if __name__ == "__main__":
    tf.app.run()
