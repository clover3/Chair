from typing import List, Tuple

from arg.counter_arg.header import Passage
from arg.counter_arg.point_counter.feature_analysis import show_features_by_svm
from arg.counter_arg.point_counter.prepare import load_data
from list_lib import lmap, right
from misc_lib import tprint
from models.baselines import svm
from task.metrics import accuracy


def main():
    train_x, train_y, dev_x, dev_y = get_data()
    tprint("training and testing")
    use_char_ngram = False
    print("Use char ngram", use_char_ngram )
    pred_svm_ngram = svm.train_svm_and_test(svm.NGramFeature(use_char_ngram, 4), train_x, train_y, dev_x)
    # pred_svm_ngram = list([random.randint(0,1) for _ in dev_y])
    result = accuracy(pred_svm_ngram, dev_y)
    print(result)


def show_features():
    train_x, train_y, dev_x, dev_y = get_data()
    show_features_by_svm(train_x, train_y)


def get_data():
    tprint("Loading data")
    train_data: List[Tuple[Passage, int]] = load_data("training")
    dev_data = load_data("validation")

    def get_texts(e: Tuple[Passage, int]) -> str:
        return e[0].text.replace("\n", " ")

    train_x: List[str] = lmap(get_texts, train_data)
    train_y: List[int] = right(train_data)
    dev_x: List[str] = lmap(get_texts, dev_data)
    dev_y: List[int] = right(dev_data)
    return train_x, train_y, dev_x, dev_y


# [{'precision': 0.920751633986928, 'recall': 0.8756798756798757, 'f1': 0.8976503385105535},
# {'precision': 0.8814814814814815, 'recall': 0.9246309246309247, 'f1': 0.9025407660219947}]

if __name__ == "__main__":
    show_features()
    # main()
#
