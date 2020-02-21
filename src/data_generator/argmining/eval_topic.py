import sys

from data_generator.argmining.eval import get_f1_score
from taskman_client.task_proxy import get_task_manager_proxy


def eval_topic(run_name, topic, split_name, condition_prefix):
    path_prefix = "./output/ukp/" + run_name
    prediction_path = path_prefix + "_" + topic
    tfrecord_path = "./data/ukp_tfrecord/{}_{}".format(split_name, topic)
    res = get_f1_score(tfrecord_path, prediction_path)
    f1_score = res["f1"]
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, f1_score, condition_prefix+topic)



if __name__ == "__main__":
    eval_topic(sys.argv[1], sys.argv[2], "dev", "F1-")