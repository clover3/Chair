from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.runner_s.run_train_3_3 import run_train3


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_v1_1_2K"
    running_option = "linear"
    feature_indices = [0, 1, 4, 5]
    save_name = f"{dataset_name}_{running_option}_0145"
    run_train3(feature_indices, dataset_name, running_option, save_name)


if __name__ == "__main__":
    main()
