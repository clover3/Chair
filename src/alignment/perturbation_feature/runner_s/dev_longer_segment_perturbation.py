from typing import List, Tuple

import numpy as np

from alignment import RelatedEvalInstance
from alignment.extract_feature import pairwise_feature_ex
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.perturbation_feature.learned_perturbation_scorer import load_pert_pred_model
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d
from bert_api import SegmentedInstance
from bert_api.segmented_instance.segmented_text import seg_to_text
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_lazy_client
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import transpose
from misc_lib import get_second, TimeEstimator
from visualize.html_visual import get_bootstrap_include_source, HtmlVisualizer, Cell


def get_search_aligner():
    nli_client = get_nli_lazy_client("localhost")
    # nli_client = get_nli_cache_client("localhost")
    model_name = 'train_v1_1_2K_linear_9'
    pert_model = load_pert_pred_model(model_name)
    shape = get_pert_train_data_shape_1d()

    def _pad_x(x):
        n_max_seg = shape[0]
        x_slice = x[:n_max_seg, :, :]
        n_pad2 = shape[0] - x_slice.shape[0]
        x_padded = np.pad(x_slice, [(0, n_pad2), (0, 0), (0, 0)])
        return x_padded

    def search_align_for_seg1_idx(si, seg1_idx):
        seg1_indices = [seg1_idx]

        def get_feature_for_seg2_indices(seg2_indices):
            si_list: List[SegmentedInstance] = pairwise_feature_ex(si.text1, si.text2,
                                                                   seg1_indices, seg2_indices)
            todo = [NLIInput(si.text2, si.text1) for si in si_list]
            logits_list = nli_client.predict(todo)
            logits_list_np = np.array(logits_list)  # [9, 3]
            return logits_list_np

        all_trials: List[Tuple[List[int], float]] = []
        for sub_seg_len in range(1, 5):
            x_list = []
            target_info = []
            for seg2_idx in si.text2.enum_seg_idx():
                if not seg2_idx + (sub_seg_len - 1) < si.text2.get_seg_len():
                    break
                seg2_indices = [seg2_idx + i for i in range(sub_seg_len)]
                logits_list_np = get_feature_for_seg2_indices(seg2_indices)
                assert logits_list_np.shape[1] == 3
                x_list.append(logits_list_np)
                target_info.append(seg2_indices)
            x = np.stack(x_list, 0)  # [seg2_len, 9, 3]
            pad_x = _pad_x(x)  # [max_seg_len, 9, 3]
            x_flat = np.reshape(pad_x, [-1, shape[1] * shape[2]])  # [max_seg_len, 9 * 3]
            X = np.expand_dims(x_flat, 0)
            y = pert_model.predict(X)  # [1, max_seg_len, 1]
            valid_seg_len = si.text2.get_seg_len() + 1 - sub_seg_len
            y = y[0, :valid_seg_len, 0]  # [valid_seg_len]
            assert len(target_info) == valid_seg_len
            for seg2_indices, score in zip(target_info, y):
                all_trials.append((seg2_indices, score))
        all_trials.sort(key=get_second, reverse=True)
        return all_trials

    return search_align_for_seg1_idx


def main():
    dataset_name = "train_head"
    search_align_for_seg1_idx = get_search_aligner()
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    html_save_name = "longer_segment.html"
    html = HtmlVisualizer(html_save_name, script_include=[get_bootstrap_include_source()])
    html.write_div_open("container")
    tokenizer = get_tokenizer()
    k = 10
    ticker = TimeEstimator(300)
    for p in problem_list[:30]:
        si = p.seg_instance
        table: List[List[Cell]] = []
        head = ['Idx', "Word"]
        head.extend(["Top {}".format(i+1) for i in range(k)])
        head_cells: List[Cell] = [Cell(c, is_head=True) for c in head]
        table.append(head_cells)
        html.write_paragraph("Text2: {}".format(seg_to_text(tokenizer, si.text2)))
        for seg1_idx in si.text1.enum_seg_idx():
            seg1_text = seg_to_text(tokenizer, si.text1.get_sliced_text([seg1_idx]))
            all_trials: List[Tuple[List[int], float]] = search_align_for_seg1_idx(si, seg1_idx)
            top_trials = all_trials[:k]

            row: List[Cell] = [Cell(str(seg1_idx)), Cell(seg1_text)]
            for seg2_indices, score in top_trials:
                slice = si.text2.get_sliced_text(seg2_indices)
                seg_text = seg_to_text(tokenizer, slice)
                cell = Cell("{0} ({1:.1f})".format(seg_text, score))
                row.append(cell)
            table.append(row)
            ticker.tick()

        table_tr = transpose(table)
        html.write_table(table_tr[1:], table_tr[0])
        html.write_paragraph("")


if __name__ == "__main__":
    main()