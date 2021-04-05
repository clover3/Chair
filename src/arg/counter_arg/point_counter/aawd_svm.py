import random

from sklearn.svm import LinearSVC

from arg.counter_arg.point_counter.feature_analysis import show_features_by_svm
from arg.counter_arg.point_counter.prepare_data import get_argu_pointwise_data
from dataset_specific.aawd.load import load_train_dev
from list_lib import lmap
from misc_lib import tprint
from models.baselines import svm
from task.metrics import accuracy, eval_3label


def main():
    # Label 0
    # [{'precision': 0.8733572281959379, 'recall': 0.9624753127057275, 'f1': 0.915753210147197},
    # Label 1
    # {'precision': 0.25, 'recall': 0.12244897959183673, 'f1': 0.1643835616438356}]
    #
    # [{'precision': 0.8733572281959379, 'recall': 0.9624753127057275, 'f1': 0.915753210147197},
    # {'precision': 0.25, 'recall': 0.12244897959183673, 'f1': 0.1643835616438356},
    # {'precision': 0.24193548387096775, 'recall': 0.078125, 'f1': 0.11811023622047244}]

    train_x, train_y, dev_x, dev_y = load_train_dev()
    tprint("training and testing")
    use_char_ngram = False
    print("Use char ngram", use_char_ngram )
    pred_svm_ngram = svm.train_svm_and_test(svm.NGramFeature(use_char_ngram, 4), train_x, train_y, dev_x)
    # pred_svm_ngram = list([random.randint(0,1) for _ in dev_y])
    result = eval_3label(pred_svm_ngram, dev_y)
    print(result)


def cross_eval():
    aawd_train_x, aawd_train_y, _, _ = load_train_dev()
    threshold = 0.8

    def translate_label(prob):
        return prob > threshold

    train_x = aawd_train_x
    train_y = aawd_train_y
    feature_extractor = svm.NGramFeature(False, 4)
    # _, _, argu_ana_dev_x, argu_ana_dev_y = get_data()
    X_train_counts = feature_extractor.fit_transform(train_x)

    eval_x = train_x
    eval_y = train_y
    x_test_count = feature_extractor.transform(eval_x)
    svclassifier = LinearSVC()
    svclassifier.fit(X_train_counts, train_y)
    test_pred = svclassifier._predict_proba_lr(x_test_count)
    disagree_prob = test_pred[:, 1]
    test_pred = lmap(translate_label, disagree_prob)
    result = accuracy(test_pred, eval_y)
    print(result)


def sanity():
    _, _, argu_ana_dev_x, argu_ana_dev_y = get_argu_pointwise_data()

    def by_random():
        return random.randint(0, 1)

    def pred1():
        return 1


    def get_accuracy(predictor):
        pred_svm_ngram = [predictor() for _ in argu_ana_dev_x]
        result = accuracy(pred_svm_ngram, argu_ana_dev_y)
        return result
    print("random", get_accuracy(by_random))
    print("pred 1", get_accuracy(pred1))


def show_features():
    train_x, train_y, dev_x, dev_y = load_train_dev()
    show_features_by_svm(train_x, train_y)


if __name__ == "__main__":
    main()
