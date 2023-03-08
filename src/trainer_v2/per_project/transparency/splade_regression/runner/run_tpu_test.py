import sys
import tensorflow as tf
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    strategy = get_strategy(True, "v2-2")
    init_checkpoint = sys.argv[1]
    with strategy.scope():
        model = tf.keras.models.load_model(init_checkpoint)






if __name__ == "__main__":
    main()