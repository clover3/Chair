import json

from cpath import at_output_dir
from estimator_helper.output_reader import join_prediction_with_info
from misc_lib import BinHistogram, get_second
from scipy_aux import logit_to_score_softmax
from tab_print import tab_print
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main():
    info_save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord.info")
    info = json.load(open(info_save_path, "r"))
    prediction_file = at_output_dir("clue_counter_arg", "ada_aawd4_clue.4000.score")
    pred_data = join_prediction_with_info(prediction_file, info)

    def bin_fn(score):
        return str(int(score * 100))

    bin = BinHistogram(bin_fn)
    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e['logits'])
        bin.add(score)

    for i in range(101):
        key = str(i)
        if key in bin.counter:
            print(key, bin.counter[key])


def aawd_pred_histogram():
    prediction_file = at_output_dir("clue_counter_arg", "ada_argu3_aawd_20000.score")
    prediction_file = at_output_dir("clue_counter_arg", "ada_aawd5_clue.4000.score")
    pred_data = EstimatorPredictionViewer(prediction_file)

    def bin_fn(score):
        return str(int(score * 1000))

    bin = BinHistogram(bin_fn)
    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e.get_vector('logits'))
        bin.add(score)

    for i in range(101):
        key = str(i)
        if key in bin.counter:
            print(key, bin.counter[key])


def show_high():
    info_save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord.info")
    info = json.load(open(info_save_path, "r"))
    # prediction_file = at_output_dir("clue_counter_arg", "ada_aawd4_clue.4000.score")
    prediction_file = at_output_dir("clue_counter_arg", "ada_aawd5_clue.4000.score")
    pred_data = join_prediction_with_info(prediction_file, info)

    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e['logits'])
        if int(score * 100) == 13:
            print(e['text'])

def print_top_k():
    k = 30
    info_save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord.info")
    info = json.load(open(info_save_path, "r"))
    prediction_file = at_output_dir("clue_counter_arg", "ada_aawd5_clue.4000.score")
    pred_data = join_prediction_with_info(prediction_file, info)

    simple_data = []

    text_set = set()
    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e['logits'])
        text = e['text']
        if text in text_set:
            continue
        text_set.add(text)
        simple_data.append((text, score))

    simple_data.sort(key=get_second, reverse=True)
    for text, score in simple_data[:k]:
        tab_print(score*100,  text)



if __name__ == "__main__":
    print_top_k()