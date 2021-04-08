from typing import List, Tuple

from arg.counter_arg.point_counter.svm_experiment.svm_wrap import SVMWrap
from arg.counter_arg_retrieval.f5.f5_docs_payload_gen import enum_f5_data
from cache import save_to_pickle, load_from_pickle
from dataset_specific.aawd.load import get_aawd_binary_train_dev
from misc_lib import tprint, two_digit_float
from tab_print import tab_print


def run_and_save():
    texts: List[str] = list(enum_f5_data())
    train_x, train_y, dev_x, dev_y = get_aawd_binary_train_dev()
    tprint("training...")
    svm = SVMWrap(train_x, train_y)

    tprint("predicting...")
    scores = svm.predict(texts)

    output: List[Tuple[str, float]] = list(zip(texts, scores))
    save_to_pickle(output, "f5_svm_aawd_prediction")


def show():
    output: List[Tuple[str, float]] = load_from_pickle("f5_svm_aawd_prediction")
    def bin_fn(score):
        return str(int(score + 0.5))

    print("{} data points ".format(len(output)))

    output.sort(key=lambda x: x[1], reverse=True)
    seen_text = set()
    output_unique = []
    for text, score in output:
        if text not in seen_text and "\n" not in text.strip():
            output_unique.append((text, score))
            seen_text.add(text)

    for text, score in output_unique[:30]:
        tab_print(two_digit_float(score),  text)

    #
    # bin = BinHistogram(bin_fn)
    # for text, score in output:
    #     bin.add(score)
    #
    # for key in bin.counter:
    #     print(key, bin.counter[key])


def debug_duplicate_different_score():
    target = "He didn't have an real credentials for this and didn't do any real work while in Niger. "
    output: List[Tuple[str, float]] = load_from_pickle("f5_svm_aawd_prediction")
    for text, score in output:
        if text == target:
            print(score)



def duplicate_check():
    texts: List[str] = list(enum_f5_data())
    print("{} data points ".format(len(texts)))
    print("unique length: ", len(set(texts)))


if __name__ == "__main__":
    # run_and_save()
    show()