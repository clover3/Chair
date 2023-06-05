from cpath import get_canonical_model_path2, get_canonical_model_path
from misc_lib import path_join
from tf_util.keras_model_to_h5 import convert_keras_model_to_h5


def main():
    tf_model_path = get_canonical_model_path2("nli14_0", "model_12500")
    dir_path = get_canonical_model_path("nli14_0")
    save_path = path_join(dir_path, "model_12500.h5py")

    convert_keras_model_to_h5(save_path, tf_model_path)


if __name__ == "__main__":
    main()