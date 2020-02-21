import os
import pickle
from collections import Counter

import math
import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr

from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import pretty_tokens
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def iter_fn(data_list):
    for iter in data_list:
        yield from iter

    return iter_fn


def clueweb_pair_prediction_list():
    path_list = []
    for j in range(3, 10):
        p = os.path.join(output_path, "clueweb_pair", "nli_prediction_{}".format(j))
        path_list.append(p)
    return path_list


def hydroponics_prediction_path_list():
    path_list = []
    for j in range(0, 30):
        p = os.path.join(output_path, "clueweb_pair_two", "abortion_hydroponics", "nli_prediction_{}".format(j))
        path_list.append(p)
    return path_list


def weather_prediction_path_list():
    path_list = []
    for j in range(30, 60):
        p = os.path.join(output_path, "clueweb_pair_two", "abortion_weather", "nli_prediction_{}".format(j))
        path_list.append(p)
    return path_list


def work(data_name):
    data, pickle_path = load_data(data_name)

    print("counting terms and predictions")
    cont, high_count, n_cont, pred_count = count_contradiction(data)
    output = analyze(cont, high_count, n_cont, pred_count)
    pickle.dump(output, open(pickle_path, "wb"))
    return output


def load_data(data_name):
    print("reading data")
    if data_name == "run_clueweb":
        path_list = clueweb_pair_prediction_list()
        data, data_len = combine_prediction_list(path_list)
        print("total of {} data".format(data_len))
        pickle_path = get_clueweb_out_pickle_path()
    elif data_name == "hydroponics":
        path_list = hydroponics_prediction_path_list()
        data, data_len = combine_prediction_list(path_list)
        pickle_path = os.path.join(output_path, data_name)
    elif data_name == "weather":
        path_list = weather_prediction_path_list()
        data, data_len = combine_prediction_list(path_list)
        pickle_path = os.path.join(output_path, data_name)
    elif data_name == "first_abortion":
        data = EstimatorPredictionViewer(os.path.join(output_path, "nli_prediction"))
        print("total of {} data".format(data.data_len))
        pickle_path = os.path.join(output_path, "abortion_nli_prediction_analysis")
    elif data_name == "rerun_abortion":
        data = EstimatorPredictionViewer(os.path.join(output_path, "abortion_contradiction"))
        print("total of {} data".format(data.data_len))
        pickle_path = os.path.join(output_path, "abortion_nli_prediction_analysis2")
    else:
        assert False
    return data, pickle_path


def view_entailment(data_name):
    data, pickle_path = load_data(data_name)
    for entry in data:
        logits = entry.get_vector("logits")
        input_ids = entry.get_vector("input_ids")
        tokens = entry.get_tokens("input_ids")

        probs = softmax(logits)
        pred = np.argmax(probs)
        if probs[0] > 0.5:
            p, h = split_p_h_with_input_ids(tokens, input_ids)
            print("P:" + pretty_tokens(p, True))
            print("H:" + pretty_tokens(h, True))
            print()


def combine_prediction_list(path_list):
    data_list = []
    data_len = 0
    for p in path_list:
        d = EstimatorPredictionViewer(p)
        data_len += d.data_len
        data_list.append(d)
    data = iter_fn(data_list)
    return data, data_len


def counter_acc(src, target):
    for key, value in src.items():
        target[key] += value

def merge_summary(dir_name, name_format):
    cont_acc = Counter()
    high_count_acc = 0
    n_cont_acc = Counter()
    pred_count_acc = Counter()
    for j in range(10, 100):
        p = os.path.join(output_path, dir_name, name_format.format(j))
        if os.path.exists(p):
            cont, high_count, n_cont, pred_count = pickle.load(open(p, "rb"))
            counter_acc(cont, cont_acc)
            counter_acc(n_cont, n_cont_acc)
            counter_acc(pred_count, pred_count_acc)
            high_count_acc += high_count
    output = analyze(cont_acc, high_count_acc, n_cont_acc, pred_count_acc)
    return output
    #pickle_path = os.path.join(output_path, dir_name + ".merged")
    #pickle.dump(output, open(pickle_path, "wb"))


def analyze(cont, high_count, n_cont, pred_count):
    def get_voca(counter):
        for key, value in counter.items():
            if value > 5:
                yield key

    print(pred_count)
    print("Selected contradiction", high_count)
    voca = set(get_voca(cont))
    ctf1 = sum(cont.values())
    voca2 = set(get_voca(n_cont))
    ctf2 = sum(n_cont.values())
    common_voca = voca.intersection(voca2)
    print("Number of common voca", len(common_voca))
    output = []
    for term in common_voca:
        p_cont = cont[term] / ctf1
        p_n_cont = n_cont[term] / ctf2
        bias = math.log(p_cont) - math.log(p_n_cont)
        output.append((term, bias, cont[term], n_cont[term]))
    output.sort(key=lambda x: x[1])
    return output


def count_contradiction(data):
    cont = Counter()
    n_cont = Counter()
    df = Counter()
    pred_count = Counter()
    high_count = 0
    for entry in data:
        logits = entry.get_vector("logits")
        input_ids = entry.get_vector("input_ids")
        tokens = entry.get_tokens("input_ids")

        probs = softmax(logits)
        pred = np.argmax(probs)
        pred_count[pred] += 1
        if probs[2] > 0.5:
            high_count += 1
            counter = cont
            if high_count < 10:
                p, h = split_p_h_with_input_ids(tokens, input_ids)
                print("P:" + pretty_tokens(p, True))
                print("H:" + pretty_tokens(h, True))
                print()

        else:
            counter = n_cont

        for t in tokens:
            if t == "[PAD]":
                break
            df[t] += 1
            counter[t] += 1
    return cont, high_count, n_cont, pred_count


def get_clueweb_out_pickle_path():
    return os.path.join(output_path, "nli_prediction_analysis")


def view(output, highlight):
    print("Total entry", len(output))
    print("More contradiction")
    for i in range(30):
        idx = len(output) - i - 1
        print(output[idx])

    print("Least contradiction")
    for i in range(30):
        print(output[i])

    bias_l = []
    tf = []
    for idx, e in enumerate(output):
        term, bias, t1, t2 = e
        bias_l.append(bias)

        tf.append((t1+t2))
        if term == "abortion" or term in highlight:
            rank = len(output) - idx
            print("Rank(more contradiction)", rank, rank / len(output))
            print(e)

    print("Pearson correlation bias-tf" , pearsonr(bias_l, tf))


def view_from_pickle(pickle_path):
    pickle_path = os.path.join(output_path, "abortion_nli_prediction_analysis")
    pickle_path = os.path.join(output_path, "clueweb_pair_summary_rerun.merged")
    output = pickle.load(open(pickle_path, "rb"))
    view(output, [""])


def see_clueweb12_13B():
    output = merge_summary("clueweb_pair_summary_rerun", "clueweb12_13B_pair_summary_{}")
    view(output, [""])


def see_rerun_abortion_10000():
    data_name = "rerun_abortion"
    output = work(data_name)
    view(output, [""])


def see_weather():
    output = work("weather")
    view(output, ["weather"])


def see_hydroponics():
    output = work("hydroponics")
    view(output, ["hydro", "##pon", "##nic"])



if __name__ == "__main__":
    view_entailment("rerun_abortion")
    #see_rerun_abortion_10000()
    #see_clueweb12_13B()
    #see_hydroponics()
    #see_weather()
    #count_and_save(int(sys.argv[1]))
