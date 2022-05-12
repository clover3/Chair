from typing import List, Tuple

from alignment.perturbation_feature.runner_s.dev_longer_segment_perturbation import get_search_aligner
from bert_api import SegmentedInstance
from bert_api.segmented_instance.segmented_text import seg_to_text, get_word_level_segmented_text_from_str, \
    SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import transpose
from visualize.html_visual import get_bootstrap_include_source, HtmlVisualizer, Cell


def get_merged_seg_text(seg_text, merge_indices):
    new_seg_indices = []
    merged_indices = []
    for i in seg_text.enum_seg_idx():
        if i in merge_indices:
            merged_indices.extend(seg_text.seg_token_indices[i])
        else:
            if merged_indices:
                new_seg_indices.append(merged_indices)
                merged_indices = []
            new_seg_indices.append(seg_text.seg_token_indices[i])
    new_hypo = SegmentedText(seg_text.tokens_ids, new_seg_indices)
    return new_hypo


def get_semantic_diff():
    search_align_for_seg1_idx = get_search_aligner()
    html_save_name = "pert_align_interactive.html"
    tokenizer = get_tokenizer()
    k = 10

    def semantic_diff(premise, hypothesis, merge_indices):
        html = HtmlVisualizer(html_save_name, script_include=[get_bootstrap_include_source()])
        html.write_div_open("container")
        prem = get_word_level_segmented_text_from_str(tokenizer, premise)
        hypo = get_word_level_segmented_text_from_str(tokenizer, hypothesis)

        new_hypo = get_merged_seg_text(hypo, merge_indices)
        si = SegmentedInstance(new_hypo, prem)
        table: List[List[Cell]] = []

        head = ['Idx', "Word"]
        head.append("Single")
        head.extend(["Top {}".format(i+1) for i in range(k)])
        head_cells: List[Cell] = [Cell(c, is_head=True) for c in head]
        table.append(head_cells)
        html.write_paragraph("Text2: {}".format(seg_to_text(tokenizer, si.text2)))
        for seg1_idx in si.text1.enum_seg_idx():
            seg1_text = seg_to_text(tokenizer, si.text1.get_sliced_text([seg1_idx]))
            all_trials: List[Tuple[List[int], float]] = search_align_for_seg1_idx(si, seg1_idx)
            single_seg_trials = [(indices, score) for indices, score in all_trials if len(indices) == 1]
            seg2_indices, score = single_seg_trials[0]
            cell = make_cell(si, seg2_indices, score)

            top_trials = all_trials[:k]
            row: List[Cell] = [Cell(str(seg1_idx)), Cell(seg1_text)]
            row.append(cell)
            for seg2_indices, score in top_trials:
                cell = make_cell(si, seg2_indices, score)
                row.append(cell)
            table.append(row)

        table_tr = transpose(table)
        html.write_table(table_tr[1:], table_tr[0])
        html.write_paragraph("")
        html.write_div_close()
        html.close()

    def make_cell(si, seg2_indices, score):
        slice = si.text2.get_sliced_text(seg2_indices)
        seg_text = seg_to_text(tokenizer, slice)
        cell = Cell("{0} ({1:.1f})".format(seg_text, score))
        return cell

    return semantic_diff


def main():
    semantic_diff = get_semantic_diff()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        indices_s = input("Merge indices: ")
        indices = list(map(int, indices_s.split()))
        semantic_diff(sent1, sent2, indices)


def debug_merge():
    tokenizer = get_tokenizer()
    hypothesis = "I did something."
    indices_s = "1 2"
    indices = list(map(int, indices_s.split()))

    hypo = get_word_level_segmented_text_from_str(tokenizer, hypothesis)
    new_hypo = get_merged_seg_text(hypo, indices)


if __name__ == "__main__":
    main()
