import sys

from data_generator.argmining.eval import get_f1_score
from taskman_client.task_proxy import get_task_manager_proxy


def main(run_name):
    path_prefix = "./output/ukp/" + run_name
    topic = "abortion"
    prediction_path = path_prefix + "_" + topic
    tfrecord_path = "./data/ukp_tfrecord_2way/dev_" + topic
    res = get_f1_score(tfrecord_path, prediction_path, 2)
    f1_score = res["f1"]
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, f1_score, "avg F-1")


if __name__ == "__main__":
    main(sys.argv[1])
