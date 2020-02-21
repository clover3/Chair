import pickle
import sys

from data_generator.argmining import ukp
from data_generator.argmining.eval import get_f1_score
from misc_lib import average
from taskman_client.task_proxy import get_task_manager_proxy


def main(base_name):
    path_prefix = "./output/ukp/" + base_name
    run_name = base_name
    all_res = {}
    for topic in ukp.all_topics:
        prediction_path = path_prefix + "_" + topic
        tfrecord_path = "./data/ukp_tfrecord/dev_" + topic
        res = get_f1_score(tfrecord_path, prediction_path)
        all_res[topic] = res

    avg_f1 = average([res["f1"] for topic, res in all_res.items()])

    for topic, res in all_res.items():
        print(topic, res["f1"])
    print("Avg F1 : ", avg_f1)
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, avg_f1, "F-1")
    pickle.dump(all_res, open("./output/ukp_score_log/"+run_name, "wb"))


if __name__ == "__main__":
    main(sys.argv[1])