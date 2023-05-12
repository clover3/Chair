import sys
import tensorflow as tf
from trainer_v2.per_project.transparency.mmp.probe.align_network import GAlignNetwork


def main():
    base_model_path = sys.argv[1]
    ranking_model = tf.keras.models.load_model(base_model_path, compile=False)
    network = GAlignNetwork(ranking_model)

    temp_model_save_path ="/tmp/model_saved"
    network.model.save(temp_model_save_path)
    model = tf.keras.models.load_model(temp_model_save_path)
    print(model)
    print("Success")


if __name__ == "__main__":
    main()