import sys

from data_generator.argmining.cloud_eval_common import get_ranking_metrics


def work(tfrecord_path, prediction_path):
    d = get_ranking_metrics(tfrecord_path, prediction_path)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])