import sys

import tensorflow as tf




def main():
    save_path = sys.argv[1]
    model = tf.keras.models.load_model(save_path)
    model.summary()



if __name__ == "__main__":
    main()