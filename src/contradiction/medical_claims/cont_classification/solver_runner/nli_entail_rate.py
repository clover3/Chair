from collections import Counter
from typing import List
import numpy as np
from contradiction.medical_claims.cont_classification.defs import ContProblem
from contradiction.medical_claims.cont_classification.path_helper import load_cont_classification_problems
from tab_print import print_table
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_predictor


def get_true_pairs_w_word(problems, word):
    target_label = 1
    pair_list = get_pairs_with_conditions(problems, target_label, word)
    return pair_list


def get_neg_pairs_w_word(problems, word):
    target_label = 0
    pair_list = get_pairs_with_conditions(problems, target_label, word)
    return pair_list


def get_pairs_with_conditions(problems, target_label, word):
    pair_list = []
    for p in problems:
        if p.label == target_label:
            pair_list.append((p.claim1_text, p.question + " " + word))
            pair_list.append((p.claim2_text, p.question + " " + word))
    return pair_list


def count_predictions(pair_list, predict_fn):
    counter = Counter()
    preds = predict_fn(pair_list)
    for probs in preds:
        pred = np.argmax(probs)
        counter[pred] += 1
    return counter


def main():
    split = "dev"
    problems: List[ContProblem] = load_cont_classification_problems(split)
    predict_fn = get_keras_nli_300_predictor()
    conditions = [
        (get_true_pairs_w_word(problems, ""), ""),
        (get_true_pairs_w_word(problems, "? yes"), "yes"),
        (get_true_pairs_w_word(problems, "? no"), "no"),

        (get_neg_pairs_w_word(problems, ""), ""),
        (get_neg_pairs_w_word(problems, "? yes"), "yes"),
        (get_neg_pairs_w_word(problems, "? no"), "no"),
    ]

    head = ["", "E", "N", "C"]
    table = [head]

    for pair_list, name in conditions:
        counter = count_predictions(pair_list, predict_fn)
        # print(counter)
        n = sum(counter.values())
        rates = ["{0:.2f}".format(counter[i] / n) for i in range(3)]
        row = [name] + rates
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()
