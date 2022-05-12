import numpy as np

from misc_lib import Averager, TEL
from ptorch.ists.eval_dataset_loader import load_dataset_from_dir
from ptorch.ists.ists_predictor import get_ists_predictor


def main():
    predictor = get_ists_predictor()
    data_dir = "C:\\work\\Code\\public_codes\\interpretable_sentence_similarity-master\\datasets\\sts_16\\train_2015_10_22.utf-8"
    eval_cases = load_dataset_from_dir(data_dir)
    recall_avg = Averager()
    for eval_case in TEL(eval_cases):
        prob_matrix = predictor.predict(eval_case.left, eval_case.right)
        n_suc = 0
        for left_idx, right_indices in eval_case.known_golds:
            prob_row = prob_matrix[left_idx]
            max_idx = np.argmax(prob_row)
            suc = (max_idx in right_indices)
            if suc:
                n_suc += 1
        recall = n_suc / len(eval_case.known_golds) if len(eval_case.known_golds) else 1
        recall_avg.append(recall)
    print("maybe recall: {0:.2f}".format(recall_avg.get_average()))


if __name__ == "__main__":
    main()