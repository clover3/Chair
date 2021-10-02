import ast
import math
import random
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import pearsonr

from abrl.cfene_log_reader import load_all_file2, drop_seen, action_sanity
from abrl.log_regression_models import convert_tensor, \
    get_model8, get_model6, get_model7, get_model5, get_model4, get_model3, get_model9, get_model10, get_model2
from cpath import at_data_dir
from list_lib import lmap, flatten, lflatten, get_max_idx, left
from misc_lib import two_digit_float
from tab_print import print_table


def log_open(file_name):
    return open(at_data_dir("rl_logs", file_name), "r")


def load_all_file() -> List[Tuple[np.array, float]]:
    def parse_discrete_action(line):
        idx, action, reward = line.split("\t")
        action = np.array(ast.literal_eval(action))
        reward = float(reward)
        return action, reward

    def parse_discrete_action_neg(line):
        action, reward = parse_discrete_action(line)
        return action, -reward

    file_list = ["episode_0.2.txt", "episode_0.5.txt", "d_1_rev.txt"]
    file_list_neg = ["dirichlet_log_1.txt"]
    all_items = []
    for file in file_list:
        items = lmap(parse_discrete_action, log_open(file))
        all_items.extend(items)

    for file in file_list_neg:
        items = lmap(parse_discrete_action_neg, log_open(file))
        all_items.extend(items)

    print("Loaded {} data points".format(len(all_items)))
    counter = Counter()
    for action, reward in all_items:
        action_sanity(action, reward)
        non_neg = [v for v in action if v >= 0]
        budget100 = int(sum(non_neg) * 100 + 0.5)
        counter[budget100] += 1

    return all_items


num_features = 7


def allocation_with_uniform(num_features, total_budget) -> List[float]:
    # draw how many to use
    n = random.randint(1, num_features)
    indices = list(range(num_features))
    # draw which to use
    indices_to_use = random.choices(indices, k=n)
    # Allocate relative budget
    d = {}
    for i in indices_to_use:
        d[i] = random.random()

    total = sum(d.values())

    budgets = []
    for j in range(num_features):
        if j in indices_to_use:
            e = d[j] / total * total_budget
        else:
            e = -1
        budgets.append(e)
    return budgets


def allocation_with_digits(num_features, total_budget) -> List[float]:
    n = random.randint(1, num_features)
    indices = list(range(num_features))
    # draw which to use
    indices_to_use = random.choices(indices, k=n)
    # Allocate relative budget

    d = {}
    for i in indices_to_use:
        d[i] = random.randint(1, 100) / 100
    total = sum(d.values())

    budgets = []
    for j in range(num_features):
        if j in indices_to_use:
            e = d[j] / total * total_budget
        else:
            e = -1
        budgets.append(e)
    return budgets



def local_search():
    budget = 0.2
    allocation = allocation_with_digits(7, budget)

    used_feature_indices = list([idx for idx, v in enumerate(allocation) if v > 0])

    step_size = 0.01
    def modify(allocation):
        idx1, idx2 = random.choices(used_feature_indices, k=2)
        new_allocation = list(allocation)
        new_allocation[idx1] += step_size
        new_allocation[idx2] -= step_size
        assert sum(new_allocation) == budget
        return new_allocation

    seen_point = set()
    get_reward = NotImplemented
    best_reward = get_reward(allocation)
    max_perturb_trial = 100
    try:
        while True:
            repeat_perturb = True
            n_trial = 0
            while repeat_perturb:
                new_allocation = modify(allocation)
                n_trial += 1
                if new_allocation not in seen_point:
                    seen_point.add(new_allocation)
                    repeat_perturb = False

                if repeat_perturb:
                    if n_trial > 10:
                        step_size = 0.5 * step_size

                    if n_trial > max_perturb_trial:
                        raise Exception()

            reward = get_reward(new_allocation)
            print(f"trying {new_allocation} got {reward}")
            if reward > best_reward:
                print("Update best")
                best_reward = reward
                allocation = new_allocation
    except:
        pass


def cdf_debug():
    stddev = 0.02
    def cdf(x):
        v = -1/2 * x * x / (stddev * stddev)
        return 1 / (stddev * math.sqrt(2*math.pi)) * math.exp(v)

    def get_p_value(diff):
        p_value = 2 * (1-cdf(tf.abs(diff)))
        return p_value

    for diff in [0.0001, 0.002, 0.01, 0.02, 0.05, 0.1]:
        print(diff, cdf(diff), get_p_value(diff))


def predict_future_actions(model, allocation_transform=None):

    if allocation_transform is None:
        def allocation_transform(a):
            return a
    scoring_batch_size = 32
    last_best = -1
    sub_model = tf.keras.Model(model.input, outputs=[model.layers[1].output])

    while True:
        action_list = []
        action_list.append([-1, -1, -1, -1, -1, 0.1, 0.1])
        for _ in range(scoring_batch_size):
            allocation = allocation_with_digits(7, 0.2)
            action_list.append(allocation)

        conv_action_list = lmap(allocation_transform, action_list)
        X = tf.data.Dataset.from_tensors(conv_action_list)
        scores = model.predict(X)
        layer1_out = sub_model.predict(X)
        idx = get_max_idx(lflatten(scores))
        new_best = scores[idx][0]
        best_action = action_list[idx]
        if new_best > last_best:
            layer1_out_str = "[" + ", ".join(lmap(two_digit_float, list(layer1_out[idx]))) + "]"
            print(best_action, new_best, conv_action_list[idx])
            last_best = new_best


def batch_pairwise_loss(pred_y, true_y):
    def get_comparison(y):
        return y - tf.transpose(y, [1, 0])

    gold_comp = get_comparison(true_y)
    pred_comp = get_comparison(pred_y)

    pairwise_hinge_loss = tf.maximum(1 - pred_comp * gold_comp, 0)
    return tf.reduce_mean(pairwise_hinge_loss)


def MSE_error_loss(pred_y, true_y):
    raw_loss = tf.keras.losses.MSE(pred_y, true_y)
    diff = pred_y - true_y
    stddev = 0.02
    normal = tfp.distributions.Normal(0, stddev)
    p_value = 2 * (1 - normal.cdf(tf.abs(diff)))
    discount_factor = 1 - p_value
    loss_by_random = tf.stop_gradient(tf.keras.losses.MSE(true_y, true_y+stddev))
    loss = raw_loss * discount_factor
    return loss


def index_and_sort(scores):
    arr = []
    for idx, s in enumerate(scores):
        arr.append((idx, s))
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr

def top_k_ap(model, dataset, dataset_batched):
    def AP(pred: List[int], gold: List[int]):
        n_pred_pos = 0
        tp = 0
        sum_prec = 0
        for idx in pred:
            n_pred_pos += 1
            if idx in gold:
                tp += 1
                sum_prec += (tp / n_pred_pos)
        return sum_prec / len(gold)

    dataset = dataset.batch(1000 * 1000)
    scores = model.predict(dataset)
    Y = list([y.numpy() for _, y in dataset.unbatch()])
    gold_rank = index_and_sort(Y)
    pred_rank = index_and_sort(flatten(scores))
    true_portion = 0.01
    n_true = int(len(gold_rank) * true_portion)
    gold_indices = left(gold_rank)[:n_true]
    return AP(left(pred_rank), gold_indices)


def eval_by_pearson(model, dataset, dataset_batched):
    dataset = dataset.batch(1000 * 1000)
    scores = model.predict(dataset)
    Y = list([y.numpy() for _, y in dataset.unbatch()])
    X = dataset.as_numpy_iterator().next()
    for x, s in zip(X, scores):
        if np.isnan(s):
            print(x, s)
    try:
        r, p = pearsonr(Y, flatten(scores))
    except ValueError:
        r = -1
    return r


def get_loss(model, dataset, dataset_batched):
    return model.evaluate(dataset_batched, verbose=0)


def run_training(train_ds, valid_ds, test_ds, model, loss_name):
    batch_size = 16
    train_ds_batched = train_ds.batch(batch_size)
    valid_ds_batched = valid_ds.batch(batch_size)
    test_ds_batched = test_ds.batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=1, verbose=1,
        mode='auto', baseline=None, restore_best_weights=False
    )

    loss_fn = {
        'MSE': MSE_error_loss,
        'pairwise': batch_pairwise_loss
    }[loss_name]

    model.compile(optimizer=optimizer,
                  loss=loss_fn)
    # loss=tf.keras.losses.MSE)

    model.fit(train_ds_batched, epochs=1000, verbose=0,
              validation_data=valid_ds_batched,
              callbacks=[early_stop])

    todo = [
        ("train", train_ds, train_ds_batched),
        ("valid", valid_ds, valid_ds_batched),
        ("test", test_ds, test_ds_batched),
    ]
    metrics = [
        ("loss", get_loss),
        ("corr", eval_by_pearson),
        ("ap", top_k_ap)
    ]

    out_d = {}
    head = [""] + left(metrics)
    rows = [head]
    for split_name, ds, ds_batched in todo:
        row = [split_name]
        for metric_name, metric in metrics:
            score = metric(model, ds, ds_batched)
            out_d[split_name, metric_name] = score
            row.append(score)
        rows.append(row)
    print_table(rows)
    return out_d


def show_top_k():
    records = drop_seen(load_all_file2())
    random.shuffle(records)
    for train_ds_size in [1000, 5000]:
        train_ds, valid_ds, test_ds = build_split_dataset(num_features, records, train_ds_size)
        test_ds = test_ds.batch(1000 * 1000)
        Y = list([y.numpy() for _, y in test_ds.unbatch()])
        gold_rank = index_and_sort(Y)
        true_portion = 20
        gold = gold_rank[:true_portion]
        print(gold)



def load_per_feature_score():
    file_path = at_data_dir("budget_allocation", "per_feature_score.txt")
    num_features = None
    rows = []
    for idx, line in enumerate(open(file_path, "r")):
        if idx == 0:
            continue
        row = list(map(float, line.split("\t")))
        rows.append(row)
        num_features = len(row) - 1

    per_feature_d = {}
    for feature_idx in range(num_features):
        scores = [(-1, 0)]
        per_feature_d[feature_idx] = scores
        for row in rows:
            budget = row[0]
            score = row[ 1 + feature_idx]
            scores.append((budget, score))
    return per_feature_d


def get_log_transformer(num_features):
    table: Dict[int, List[float, float]] = load_per_feature_score()
    eps = 1e-6

    def transform(tag) -> Tuple[List[float], float]:
        action, reward = tag
        new_action = []
        for i in range(num_features):
            cur_budget = action[i]
            feature_score_map = table[i]
            p_reward = get_matching_feature_score(cur_budget, feature_score_map)
            new_action.append(p_reward)
        return new_action, reward

    def get_matching_feature_score(cur_budget, feature_score_map):
        for budget, p_reward in feature_score_map:
            if cur_budget < budget + eps:
                return p_reward
        raise IndexError

    return transform


def run_model9_10():
    max_budget = 0.2
    records = drop_seen(load_all_file2())
    print("Loaded {} data points".format(len(records)))
    num_features = 7
    record_transform_fn = get_log_transformer(num_features)
    mapped_records = lmap(record_transform_fn, records)
    train_ds_size = 10000
    train_ds, valid_ds = build_split_dataset(num_features, mapped_records, train_ds_size)
    model = get_model9(num_features, max_budget)
    best_score = run_training(train_ds, valid_ds, model, 'pairwise')

    def record_transform_fn_wo_reward(action):
        new_action, _ = record_transform_fn((action, 0))
        return new_action

    predict_future_actions(model, record_transform_fn_wo_reward)


def build_split_dataset(num_features, records, train_ds_size=None):
    dataset: tf.data.Dataset = convert_tensor(records, num_features)
    if train_ds_size is not None:
        full_ds_size = len(records)
        valid_ds_size = int((full_ds_size - train_ds_size)/2)
    else:
        full_ds_size = len(records)
        train_ds_size = int(0.64 * full_ds_size)
        valid_ds_size = int(0.16 * full_ds_size)
    test_ds_size = full_ds_size - valid_ds_size - train_ds_size

    train_ds = dataset.take(train_ds_size)
    remaining = dataset.skip(train_ds_size)
    valid_ds = remaining.take(valid_ds_size)
    test_ds = remaining.skip(valid_ds_size)
    print("train/val/test size: {}/{}/{}".format(train_ds_size, valid_ds_size, test_ds_size))
    return train_ds, valid_ds, test_ds


def parameter_summary():
    for model_get_method in [get_model2, get_model3, get_model4, get_model5, get_model6, get_model7, get_model8,
                             get_model9, get_model10]:
        model = model_get_method(num_features, 0.2)
        model.summary()


def run_multiple_models():
    print("Init")
    num_features = 7
    max_budget = 0.2
    records = drop_seen(load_all_file2())
    random.shuffle(records)
    record_transform_fn = get_log_transformer(num_features)
    mapped_records = lmap(record_transform_fn, records)

    print("Loaded {} data points".format(len(records)))
    test_loss_list_list = []
    test_corr_list_list = []
    for train_ds_size in [100, 1000, 5000]:
        print("Train ds size=", train_ds_size)
        train_ds, valid_ds, test_ds = build_split_dataset(num_features, records, train_ds_size)
        train_ds_c, valid_ds_c, test_ds_c = build_split_dataset(num_features, mapped_records, train_ds_size)

        loss_name = 'pairwise'
        # loss_name = "MSE"
        scores_d_list = []
        for model_get_method in [get_model2, get_model3, get_model4, get_model5, get_model6, get_model7, get_model8]:
            try:
                model = model_get_method(num_features, max_budget)
                scores_d = run_training(train_ds, valid_ds, test_ds, model, loss_name)
                scores_d_list.append(scores_d)
            except ValueError as e:
                print(e)
                pass

        for model_get_method in [get_model9, get_model10]:
            try:
                model = model_get_method(num_features, max_budget)
                scores_d = run_training(train_ds_c, valid_ds_c, test_ds_c, model, loss_name)
                scores_d_list.append(scores_d)
            except ValueError:
                print(e)
                pass

        rows = []
        for split in ["train", "valid", "test"]:
            row = [split]
            for j in range(len(scores_d_list)):
                for metric in ["loss", "corr", "ap"]:
                    row.append(scores_d_list[j][split, metric])
            rows.append(row)

    print_table(rows)


def main():
    print("Init")
    num_features = 7
    max_budget = 0.2
    records = drop_seen(load_all_file2())
    random.shuffle(records)
    print("Loaded {} data points".format(len(records)))
    train_ds_size = 5000
    print("Train ds size=", train_ds_size)
    record_transform_fn = get_log_transformer(num_features)
    mapped_records = lmap(record_transform_fn, records)
    train_ds, valid_ds, test_ds = build_split_dataset(num_features, records, train_ds_size)
    train_ds_c, valid_ds_c, test_ds_c = build_split_dataset(num_features, mapped_records, train_ds_size)

    loss_name = "pairwise"
    model = get_model9(num_features, max_budget)
    d = run_training(train_ds_c, valid_ds_c, test_ds_c, model, loss_name)
    print(d)

    def record_transform_fn_wo_reward(action):
        new_action, _ = record_transform_fn((action, 0))
        return new_action

    predict_future_actions(model, record_transform_fn_wo_reward)

if __name__ == "__main__":
    main()
