import math
import sys
from typing import List
import numpy as np

from cpath import get_canonical_model_path2
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from misc_lib import two_digit_float
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.inference import InferenceHelperSimple
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.evidence_selector.calc_candidate_evidences import EvidenceCompare
from trainer_v2.evidence_selector.evidence_candidates import EvidencePair, PHSegmentedPairParser, ScoredEvidencePair
from trainer_v2.train_util.arg_flags import flags_parser


def cross_entropy(pred_prob, gold_prob):
    eps = 0.0000001
    v = - pred_prob * math.log(gold_prob) - (1-pred_prob) * math.log(1 - gold_prob + eps)
    return v


def main(args):
    # Load model
    model_path = get_canonical_model_path2("nli_ts_run98_0", "model_25000")
    run_config = get_run_config_for_predict(args)
    model = load_local_decision_model(model_path)

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, ModelConfig600_3(), is_for_training)

    parser = PHSegmentedPairParser(300)
    evidence_compare = EvidenceCompare(parser)
    predict_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    tokenizer = get_tokenizer()

    # Lower the better
    def evidence_score_core(err, num_del_tokens, max_num_tokens):
        num_tokens = max_num_tokens - num_del_tokens
        tolerance = 0.05
        return max(tolerance, err) + tolerance * (num_tokens / max_num_tokens)

    inference = InferenceHelperSimple(model)
    batch_prediction_enum = inference.enum_batch_prediction(predict_dataset)

    for group in evidence_compare.enum_grouped(batch_prediction_enum):
        base_item: ScoredEvidencePair = group[0]
        base_pair = base_item.pair
        base_pair: EvidencePair = base_pair
        print("\n")
        print("Prem: ", ids_to_text(tokenizer, base_pair.p_tokens))
        # print("Hypo1: ", ids_to_text(tokenizer, base_pair.h1))
        # print("Hypo2: ", ids_to_text(tokenizer, base_pair.h2))
        base_l_y = base_item.l_y

        other_items: List[ScoredEvidencePair] = group[1:]
        for prem_i in [0, 1]:
            print("Hypo{}: {}".format(prem_i+1, ids_to_text(tokenizer, [base_pair.h1, base_pair.h2][prem_i])))
            print("Prem {}".format(prem_i+1))
            print("Local label", base_l_y[prem_i])

            def evidence_score(item: ScoredEvidencePair) -> float:
                pair = item.pair
                del_indices = [pair.p_del_indices1, pair.p_del_indices2]
                err = np.sum(np.abs(base_l_y[prem_i] - item.l_y[prem_i]))
                score = evidence_score_core(err, len(del_indices[prem_i]), len(pair.p_tokens))
                return score

            other_items.sort(key=evidence_score)

            print("Score Err Len_Penalty, [Probabilities]")
            for item in other_items:
                p_cur = list(base_pair.p_tokens)
                for j in [item.pair.p_del_indices1, item.pair.p_del_indices2][prem_i]:
                    p_cur[j] = parser.mask_id
                pair = item.pair
                del_indices = [pair.p_del_indices1, pair.p_del_indices2]
                err = np.sum(np.abs(base_l_y[prem_i] - item.l_y[prem_i]))

                loss_arr = []
                for label_i in range(3):
                    loss = cross_entropy(item.l_y[prem_i][label_i], base_l_y[prem_i][label_i])
                    loss_arr.append(loss)
                ce_err = np.mean(loss_arr)
                score = evidence_score_core(ce_err, len(del_indices[prem_i]), len(pair.p_tokens))
                max_num_tokens = len(pair.p_tokens)
                num_del_tokens = len(del_indices[prem_i])
                num_tokens = max_num_tokens - num_del_tokens

                len_penalty = 0.05 * (num_tokens / max_num_tokens)
                probs = ", ".join(map(two_digit_float, item.l_y[prem_i]))
                print("{0:.2f} {1:.2f} {2:.2f} [{3}] {4}".format(score, ce_err, len_penalty,
                                                               probs,
                                                               ids_to_text(tokenizer, p_cur)))






if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
