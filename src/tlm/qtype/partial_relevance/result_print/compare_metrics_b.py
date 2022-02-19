from collections import Counter
from typing import List, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedBinaryAnswer, rei_to_text
from tlm.qtype.partial_relevance.eval_metric.meta_common import better_fn_d
from tlm.qtype.partial_relevance.eval_score_dp_helper import load_eval_result_b_single
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer
from tlm.qtype.partial_relevance.result_print.answer_load_util import get_index_answer_dict
from visualize.html_visual import HtmlVisualizer, Cell


def get_answer_indices_b(target_idx, p, table):
    answer_indices = []
    for i in p.seg_instance.text2.enum_seg_idx():
        score = table[target_idx][i]
        if not type(score) == int:
            raise TypeError
        if int(score):
            answer_indices.append(i)
    return answer_indices


def load_scores(dataset, method_list, metric_list):
    score_d_d: Dict[Tuple[str, str], Dict[str, float]] = {}
    for metric in metric_list:
        for method in method_list:
            eval_res: List[Tuple[str, float]] = get_score_for_method(dataset, method, metric)
            score_d_d[metric, method] = dict(eval_res)
    return score_d_d


def get_score_for_method(dataset, method, metric):
    run_name = "{}_{}_{}".format(dataset, method, metric)
    eval_res = load_eval_result_b_single(run_name)
    return eval_res


def get_win_counts(method_list, metric_list, problems, score_d_d):
    counter = Counter()
    for p in problems:
        best_method_per_metric = get_best_method_per_metric(p, score_d_d, method_list, metric_list)
        for metric in metric_list:
            counter[metric, best_method_per_metric[metric]] += 1
    return counter


def get_best_method_per_metric(p, score_d_d, method_list, metric_list):
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
    return best_method_per_metric


def main():
    dataset = "dev_sent"
    method_list = ["exact_match", "random"]
    metric_list = ["ps_replace_precision", "ps_replace_recall",
                   "ps_deletion_precision", "ps_deletion_recall",
                   "attn"]
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    target_idx = 1
    tokenizer = get_tokenizer()

    score_d_d: Dict[Tuple[str, str], Dict[str, float]] = load_scores(dataset, method_list, metric_list)
    related_answer_load_fn = load_binary_related_eval_answer
    answer_d: Dict[str, Dict[str, RelatedBinaryAnswer]] = get_index_answer_dict(dataset, method_list,
                                                                                related_answer_load_fn)
    html = HtmlVisualizer("align_compare_b.html")

    for p in problems:
        best_method_per_metric = get_best_method_per_metric(p, score_d_d, method_list, metric_list)
        html.write_paragraph(rei_to_text(tokenizer, p))
        html.write_paragraph("< Preference > ")

        table = []
        for metric in metric_list:
            row = [Cell(metric), Cell(best_method_per_metric[metric])]
            table.append(row)
        html.write_table(table)

        html.write_paragraph("< Method Output > ")
        for method in method_list:
            answer = answer_d[method][p.problem_id]
            try:
                answer_indices = get_answer_indices_b(target_idx, p, answer.score_table)
                inst = p.seg_instance
                s_list: List[str] = [inst.text2.get_segment_tokens_rep(tokenizer, i) for i in answer_indices]
                s = method + ":" + " ".join(s_list)
                html.write_paragraph(s)
            except TypeError:
                print(method)
                raise

        # answer = answer_d["gradient"][p.problem_id]
        # write_table(tokenizer, html, p.seg_instance, answer.score_table)

    counter = get_win_counts(method_list, metric_list, problems, score_d_d)
    for metric in metric_list:
        for method in method_list:
            html.write_paragraph(f"{metric}\t{method}\t{counter[metric, method]}")


if __name__ == "__main__":
    main()
