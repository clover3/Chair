from functools import partial

from data_generator.argmining.eval import load_tfrecord, load_preditions, compare
from evaluation import AP_from_binary
from misc_lib import left, average
from task.metrics import eval_3label


def get_pred_tfrecord_path(run_name, topic):
    path_prefix = "./output/ukp/" + run_name
    prediction_path = path_prefix + "_" + topic
    tfrecord_path = "./data/ukp_tfrecord/dev_" + topic
    return prediction_path, tfrecord_path


def join_label(tfrecord, predictions):
    for record, pred in zip(tfrecord, predictions):
        input_ids_r, label_ids = record
        input_ids_p, logits = pred
        yield label_ids, logits


def get_ranking_metrics(tfrecord_path, prediction_path):
    tfrecord = list(load_tfrecord(tfrecord_path))
    predictions = list(load_preditions(prediction_path))

    label_and_prediction = list(join_label(tfrecord, predictions))

    golds, preds = zip(*compare(tfrecord, predictions))
    golds = golds[:len(preds)]

    for result in eval_3label(preds, golds):
        print(result)

    def get_score(label_idx, entry):
        label, logits = entry
        return logits[label_idx]

    ap_list = []
    for target_label in [0, 1, 2]:
        print("Label : ", target_label)
        key_fn = partial(get_score, target_label)
        label_and_prediction.sort(key=key_fn, reverse=True)
        labels = left(label_and_prediction)

        correctness_list = list([l == target_label for l in labels])
        num_gold = sum(correctness_list)

        ap = AP_from_binary(correctness_list, num_gold)
        ap_list.append(ap)
        print("AP: ", ap)

        k_list = [1, 5, 10, 100]
        print("P at {}".format(k_list), end="\t")
        show_all_p_at_k(correctness_list, label_and_prediction)

    MAP = average(ap_list)
    return {"MAP": MAP}


def show_all_p_at_k(correctness_list, label_and_prediction):
    prev_c = -1
    ed = len(correctness_list)
    for k in range(1, ed):
        c = sum(correctness_list[:k])
        if c != prev_c:
            print(c)
            p_at_k = c / k
            print("{}/{}\t{} - {}".format(c, k, p_at_k, label_and_prediction[k]))
        prev_c = c
    print("")