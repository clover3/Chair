from arg.counter_arg.point_counter.feature_analysis import show_features_by_svm
from arg.counter_arg.point_counter.prepare_data import get_argu_pointwise_data
from misc_lib import tprint
from models.baselines import svm
from task.metrics import accuracy


def main():
    train_x, train_y, dev_x, dev_y = get_argu_pointwise_data()
    tprint("training and testing")
    use_char_ngram = False
    print("Use char ngram", use_char_ngram )
    pred_svm_ngram = svm.train_svm_and_test(svm.NGramFeature(use_char_ngram, 4), train_x, train_y, dev_x)
    # pred_svm_ngram = list([random.randint(0,1) for _ in dev_y])
    result = accuracy(pred_svm_ngram, dev_y)
    print(result)


def show_features():
    train_x, train_y, dev_x, dev_y = get_argu_pointwise_data()
    show_features_by_svm(train_x, train_y)


# [{'precision': 0.920751633986928, 'recall': 0.8756798756798757, 'f1': 0.8976503385105535},
# {'precision': 0.8814814814814815, 'recall': 0.9246309246309247, 'f1': 0.9025407660219947}]

if __name__ == "__main__":
    show_features()
    # main()
#
