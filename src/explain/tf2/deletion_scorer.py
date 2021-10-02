import os
import pickle
from typing import NamedTuple, List, Dict, Tuple, Callable

from data_generator.tokenizer_wo_tf import get_tokenizer
from visualize.html_visual import HtmlVisualizer, Cell


class TokenExEntry(NamedTuple):
    input_ids: List[int]
    base_score: float
    contribution: Dict[int, float]
    data_id: int
    # logits need softmax to be transformed
    base_logits: List[float]
    case_logits_d: Dict[int, List[float]]


# dir_path has a number of estimator_outputs for different shift, filename should be str(shift)
def summarize_deletion_score_batch8(dir_path: str, n_trial: int,
                                    shift_list: List[int],
                                    signal_fn: Callable[[List[float]], float]) -> List[TokenExEntry]:
    batch_size = 8
    return summarize_deletion_score(dir_path, n_trial, batch_size, shift_list, signal_fn)


def summarize_deletion_score(dir_path: str, n_trial: int, batch_size: int,
                             shift_list: List[int],
                             signal_fn: Callable[[List[float]], float]) -> List[TokenExEntry]:

    data_list: List[List[Dict]] = []
    for shift in shift_list:
        save_path = os.path.join(dir_path, str(shift))
        data = pickle.load(open(save_path, "rb"))
        data_list.append(data)
    num_batch = len(data_list[0])
    num_real_data = batch_size * num_batch
    data_bucket, global_data_idx_to_data_id, input_id_dict = group_results(batch_size,
                                                                           data_list,
                                                                           n_trial,
                                                                           num_batch,
                                                                           num_real_data,
                                                                           shift_list)

    summarized_result: List[TokenExEntry] = []
    for global_data_idx in range(num_real_data):
        data_id = global_data_idx_to_data_id[global_data_idx]
        input_ids = input_id_dict[global_data_idx]
        per_real_dp_data = data_bucket[global_data_idx]
        base_logit = None
        for del_location, case_logit in per_real_dp_data:
            if del_location is None:
                base_logit = case_logit
                break
        prob = signal_fn(base_logit)

        contribution: Dict[int, float] = {}
        case_logits_d: Dict[int, List[float]] = {}
        for del_location, case_logit in per_real_dp_data:
            if del_location is not None:
                case_prob: float = signal_fn(case_logit)
                score_change: float = case_prob - prob
                contribution[del_location] = score_change
                case_logits_d[del_location] = case_logit
        summarized_result.append(TokenExEntry(input_ids, prob, contribution, int(data_id), base_logit, case_logits_d))
    return summarized_result


def group_results(batch_size,
                  data_list: List[List[Dict]],
                  n_trial,
                  num_batch,
                  num_real_data,
                  shift_list):
    Logits = List[float]
    data_bucket: List[List[Tuple[int, Logits]]] = [list() for _ in range(num_real_data)]
    input_id_dict = {}
    global_data_idx_to_data_id = {}

    def check_assert(data_id, global_data_idx):
        if global_data_idx not in global_data_idx_to_data_id:
            global_data_idx_to_data_id[global_data_idx] = data_id
        else:
            assert global_data_idx_to_data_id[global_data_idx] == data_id

    for batch_idx in range(num_batch):
        for shift_idx, shift in enumerate(shift_list):
            cur_data = data_list[shift_idx][batch_idx]
            logits = cur_data["logits"]  # [batch_size * (n+1),
            input_ids = cur_data["input_ids"]
            data_id_l = cur_data["data_id"]

            batch_size_temp = len(input_ids)
            assert batch_size_temp == batch_size
            batch_size = len(input_ids)
            row_idx = 0
            for inside_batch_idx in range(batch_size):
                global_data_idx = batch_idx * batch_size + inside_batch_idx
                check_assert(data_id_l[inside_batch_idx], global_data_idx)
                input_id_dict[global_data_idx] = input_ids[inside_batch_idx]
                base_logit: Logits = logits[row_idx]
                data_bucket[global_data_idx].append((None, base_logit))
                row_idx += 1
                for del_idx in range(n_trial):
                    del_location = shift + del_idx
                    case_logit: Logits = logits[row_idx]
                    row_idx += 1
                    data_bucket[global_data_idx].append((del_location, case_logit))
    return data_bucket, global_data_idx_to_data_id, input_id_dict


def show(out_file_name, summarized_table: List[TokenExEntry]):
    html = HtmlVisualizer(out_file_name)
    tokenizer = get_tokenizer()
    num_print = 0
    for input_ids, prob, contributions in summarized_table:
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        html.write_paragraph("Score : {}".format(prob))
        cells = []
        max_change = 0
        for idx in range(len(input_ids)):
            token = tokens[idx]
            if token == "[PAD]":
                break
            if idx in contributions:
                raw_score = contributions[idx]
                max_change = max(abs(raw_score), max_change)

                score = abs(raw_score) * 100
                color = "R" if raw_score > 0 else "B"
                c = Cell(token, highlight_score=score, target_color=color)
            else:
                c = Cell(token, highlight_score=150, target_color="Gray")
            cells.append(c)

        if max_change < 0.05:
            pass
        else:
            html.multirow_print(cells, 30)
            num_print += 1

    print("printed {} of {}".format(num_print, len(summarized_table)))