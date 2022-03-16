import json
from collections import Counter
from typing import List, Tuple, Dict

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from list_lib import l_to_map, dict_value_map, index_by_fn, flatten
from misc_lib import find_min_idx
from tlm.qtype.partial_relevance.attention_based.runner.save_detail_score import get_attn_detail_save_path
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, \
    ContributionSummary, rei_to_text
from tlm.qtype.partial_relevance.eval_metric.attn_mask_utils import get_drop_mask
from tlm.qtype.partial_relevance.eval_metric.meta_common import better_fn_d
from tlm.qtype.partial_relevance.eval_score_dp_helper import load_eval_result_r
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer
from visualize.html_visual import HtmlVisualizer, Cell


def get_score_for_method(dataset, method, metric):
    run_name = "{}_{}_{}".format(dataset, method, metric)
    eval_res = load_eval_result_r(run_name)
    return eval_res


def show_brevity_preference():
    p = get_attn_detail_save_path("dev_sent", "gradient")
    info_d: Dict[str, List[Dict]] = json.load(open(p, "r"))
    keep_portion_list = [i / 10 for i in range(10)]

    for problem_id, items in info_d.items():
        for e in items:
            e['score'] = e['fidelity'] + e['brevity']
        min_idx = find_min_idx(lambda e: e['score'], items)
        print(keep_portion_list[min_idx])


def get_index_answer_dict(dataset, method_list):
    def load_answers(method):
        return load_related_eval_answer(dataset, method)

    answer_d_raw: Dict[str, List[RelatedEvalAnswer]] = l_to_map(load_answers, method_list)

    def index_answer_list(l: List[RelatedEvalAnswer]) -> Dict[str, RelatedEvalAnswer]:
        return index_by_fn(lambda a: a.problem_id, l)

    answer_d: Dict[str, Dict[str, RelatedEvalAnswer]] = dict_value_map(index_answer_list, answer_d_raw)
    return answer_d


def write_table(tokenizer, html: HtmlVisualizer, seg: SegmentedInstance, table: List[List[float]]):
    max_score = max(flatten(table))
    min_score = min(flatten(table))
    norm_cap = max(abs(max_score), abs(min_score))

    def get_cell(score):
        highlight_score = 200 * abs(score) / norm_cap
        color = "B" if score > 0 else "R"
        return Cell("", highlight_score, target_color=color)

    def get_words(text) -> List[str]:
        return [ids_to_text(tokenizer, text.get_tokens_for_seg(i)) for i in text.enum_seg_idx()]

    seg1_tokens = get_words(seg.text1)
    seg2_tokens = get_words(seg.text2)
    head = [Cell("")]
    head.extend(map(Cell, seg2_tokens))
    html_table = [head]
    for idx, row in enumerate(table):
        html_row = [Cell(seg1_tokens[idx])]
        html_row.extend(map(get_cell, row))
        html_table.append(html_row)
    html.write_table(html_table)


def main():
    dataset = "dev_sent"
    method_list = ["exact_match", "gradient"]
    metric_list = ["ps_replace_precision", "ps_replace_recall", "attn_brevity"]
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    p_dict = index_by_fn(lambda p: p.problem_id, problems)
    tokenizer = get_tokenizer()
    p = get_attn_detail_save_path(dataset, "gradient")
    info_d: Dict[str, List[Dict]] = json.load(open(p, "r"))
    target_idx = 1

    answer_d: Dict[str, Dict[str, RelatedEvalAnswer]] = get_index_answer_dict(dataset, method_list)

    def get_drop_mask_used(problem_id, contrib: ContributionSummary):
        e_list: List[Dict] = info_d[problem_id]
        assert len(e_list) == 10
        p: RelatedEvalInstance = p_dict[problem_id]
        for e in e_list:
            f = e['fidelity']
            b = e['brevity']
            e['score'] = f + b

        min_idx = find_min_idx(lambda x: x['score'], e_list)
        used_brevity = e_list[min_idx]['brevity']
        # print(e_list)
        # print("used_brevity", used_brevity)

        keep_portion_list = [i / 10 for i in range(10)]
        drop_k_list = [1-v for v in keep_portion_list]
        total_items = p.seg_instance.text1.get_seg_len() * p.seg_instance.text2.get_seg_len()
        drop_rate_brevity_list = []
        for drop_rate in drop_k_list:
            keep_portion = 1 - drop_rate
            n_used_item_in_target_seg = int(keep_portion * p.seg_instance.text2.get_seg_len())
            n_drop_item = p.seg_instance.text2.get_seg_len() - n_used_item_in_target_seg
            n_used_item = total_items - n_drop_item
            brevity = n_used_item / total_items
            drop_rate_brevity_list.append((drop_rate, brevity))

        # print(drop_rate_brevity_list)
        def dist(i):
            _, brevity = drop_rate_brevity_list[i]
            return abs(used_brevity - brevity)
        used_idx = find_min_idx(dist, range(10))
        guessed_brevity = drop_rate_brevity_list[used_idx][1]
        # print(f"used_idx={used_idx} guessed_brevity={guessed_brevity}")
        # print(abs(guessed_brevity - used_brevity))
        assert abs(guessed_brevity - used_brevity) < 1e-7
        given_drop_rate, _ = drop_rate_brevity_list[used_idx]
        mask_d = get_drop_mask(contrib, given_drop_rate, p.seg_instance, target_idx)
        return mask_d

    def get_method_prediction_str(p: RelatedEvalInstance, method):
        answer_indices = get_answer_indices(p, method)
        inst = p.seg_instance
        s_list: List[str] = [inst.text2.get_segment_tokens_rep(tokenizer, i) for i in answer_indices]
        return method + ":" + " ".join(s_list)

    def get_answer_indices(p, method):
        a: RelatedEvalAnswer = answer_d[method][p.problem_id]
        table = a.contribution.table
        if method == "exact_match":
            answer_indices = get_answer_indices_b(p, table)
        elif method == "gradient":
            mask_d = get_drop_mask_used(a.problem_id, a.contribution)
            drop_indices = []
            for key, v in mask_d.items():
                seg1_idx, seg2_idx = key
                if seg1_idx == target_idx and v:
                    drop_indices.append(seg2_idx)
            answer_indices = [i for i in p.seg_instance.text2.enum_seg_idx() if i not in drop_indices]
        else:
            assert False
        return answer_indices

    def get_answer_indices_b(p, table):
        answer_indices = []
        for i in p.seg_instance.text2.enum_seg_idx():
            score = table[target_idx][i]
            assert type(score) == int
            if int(score):
                answer_indices.append(i)
        return answer_indices

    score_d_d: Dict[Tuple[str, str], Dict[str, float]] = {}
    for metric in metric_list:
        for method in method_list:
            eval_res: List[Tuple[str, float]] = get_score_for_method(dataset, method, metric)
            score_d_d[metric, method] = dict(eval_res)

    html = HtmlVisualizer("align_compare.html")
    counter = Counter()
    for p in problems:
        print("---------")
        best_method_per_metric = {}
        for metric in metric_list:
            best_method = method_list[0]
            better_fn = better_fn_d[metric]
            score_per_method = {}
            for method in method_list:
                score: float = score_d_d[metric, method][p.problem_id]
                score_per_method[method] = score
                if better_fn(score_per_method[best_method], score):
                    best_method = method

            best_method_per_metric[metric] = best_method
        html.write_paragraph(rei_to_text(tokenizer, p))
        html.write_paragraph("< Preference > ")

        table = []
        for metric in metric_list:
            counter[metric, best_method_per_metric[metric]] += 1
            row = [Cell(metric), Cell(best_method_per_metric[metric])]
            table.append(row)
        html.write_table(table)

        html.write_paragraph("< Method Output > ")
        for method in method_list:
            html.write_paragraph(get_method_prediction_str(p, method))

        answer = answer_d["gradient"][p.problem_id]
        write_table(tokenizer, html, p.seg_instance, answer.contribution.table)

    for metric in metric_list:
        for method in method_list:
            html.write_paragraph(f"{metric}\t{method}\t{counter[metric, method]}")


if __name__ == "__main__":
    main()
