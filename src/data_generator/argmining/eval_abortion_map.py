import sys

from data_generator.argmining.cloud_eval_common import get_pred_tfrecord_path, get_ranking_metrics
from taskman_client.task_proxy import get_task_manager_proxy


def abortion_eval(run_name):
    topic = "abortion"
    prediction_path, tfrecord_path = get_pred_tfrecord_path(run_name, topic)
    d = get_ranking_metrics(tfrecord_path, prediction_path)
    MAP = d['MAP']
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, MAP, "MAP")


if __name__ == "__main__":
    abortion_eval(sys.argv[1])