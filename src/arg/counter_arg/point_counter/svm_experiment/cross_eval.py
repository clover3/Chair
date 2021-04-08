from collections import Counter

from arg.counter_arg.point_counter.prepare_data import get_argu_pointwise_data
from arg.counter_arg.point_counter.sklearn_metrics import get_ap
from arg.counter_arg.point_counter.svm_experiment.svm_wrap import SVMWrap
from dataset_specific.aawd.load import get_aawd_binary_train_dev
from misc_lib import print_dict_tab
from task.metrics import accuracy


def cross_eval(source, target):
    dataset_name_list = [source, target]
    data_get_method = {
        'aawd': get_aawd_binary_train_dev,
        'argu': get_argu_pointwise_data
    }
    all_splits = {}
    for dataset_name in dataset_name_list:
        train_x, train_y, dev_x, dev_y = data_get_method[dataset_name]()
        all_splits[(dataset_name, 'train')] = train_x, train_y
        all_splits[(dataset_name, 'dev')] = dev_x, dev_y

    train_x, train_y = all_splits[(source, 'train')]
    print(Counter(train_y))
    svm = SVMWrap(train_x, train_y)

    def predict_and_eval(dataset_name, split):
        eval_x, eval_y = all_splits[(dataset_name, split)]
        test_pred = svm.predict(eval_x)
        return {
                    'acc': accuracy(test_pred > 0, eval_y),
                    'ap': get_ap(eval_y, test_pred)
                }

    print("source -> target: {} -> {}".format(source, target))
    print("in domain")
    print_dict_tab(predict_and_eval(source, 'dev'))
    print("out domain")
    print_dict_tab(predict_and_eval(target, 'dev'))


def main():
    argu = 'argu'
    aawd = 'aawd'
    cross_eval(argu, aawd)
    cross_eval(aawd, argu)


if __name__ == "__main__":
    main()
