import json
import os
from typing import List, Tuple, Any, Dict

from alignment.data_structure import ContributionSummary
from alignment.data_structure.eval_data_structure import Alignment2D, \
    join_p_withother
from alignment.data_structure.print_helper import rei_to_text
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from list_lib import index_by_fn
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricWCIF
from tlm.qtype.partial_relevance.get_policy_util import get_eval_policy_wc
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer, \
    load_related_eval_answer, \
    parse_related_eval_answer_from_json


def get_one_hot_contribution(p: RelatedEvalInstance, idx1: int, idx2: int) -> Alignment2D:
    inst = p.seg_instance
    np_array = inst.get_drop_mask(idx1, idx2)
    table: List[List[float]] = p.seg_instance.score_d_to_table(np_array)
    return Alignment2D(p.problem_id, ContributionSummary(table))


def partial_related_search(
        p_c_list: List[Tuple[RelatedEvalInstance,
                             ComplementSearchOutput]],
        eval_policy: EvalMetricWCIF,
        preserve_idx) -> List[Alignment2D]:

    def get_predictions_for_case(a_p_c: Tuple[Alignment2D,
                                              RelatedEvalInstance,
                                              ComplementSearchOutput]):
        a, p, c = a_p_c
        return eval_policy.get_predictions_for_case(p, a, c)

    per_pc = []
    for p, c in p_c_list:
        dmc_list: List[int] = list(range(p.seg_instance.text2.get_seg_len()))
        dmc_paired_future = []
        for dmc in dmc_list:
            answer: Alignment2D = get_one_hot_contribution(p, preserve_idx, dmc)
            future = get_predictions_for_case((answer, p, c))
            dmc_paired_future.append((dmc, future))
        per_pc.append((p, c, dmc_paired_future))

    eval_policy.do_duty()

    def summarize(p: RelatedEvalInstance, dmc_paired: List[Tuple[Any, float]]) -> Alignment2D:
        score_d = dict(dmc_paired)
        scores = [score_d[idx] for idx in range(p.seg_instance.text2.get_seg_len())]
        cs = ContributionSummary.from_single_array(scores, preserve_idx, 2)
        return Alignment2D(p.problem_id, cs)

    output: List[Alignment2D] = []
    for p, c, dmc_paired_future in per_pc:
        dmc_paired = [(dmc, eval_policy.convert_future_to_score(future)) for dmc, future in dmc_paired_future]
        a: Alignment2D = summarize(p, dmc_paired)
        output.append(a)
    return output


def join_pc(problems: List[RelatedEvalInstance],
            complements: List[ComplementSearchOutput]) -> List[Tuple[RelatedEvalInstance, ComplementSearchOutput]]:
    pid_to_c: Dict[str, ComplementSearchOutput] = index_by_fn(lambda e: e.problem_id, complements)
    output = []
    for p in problems:
        c = pid_to_c[p.problem_id]
        output.append((p, c))
    return output


def run_search(dataset, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    problems = [p for p in problems if p.score >= 0.5]
    problems = problems[:10]
    target_idx = 0
    dataset = dataset + "_ten"
    eval_policy = get_eval_policy_wc(policy_name, model_interface, target_idx)
    complements = load_complements()
    pc_list: List[Tuple[RelatedEvalInstance, ComplementSearchOutput]] = join_pc(problems, complements)
    answers: List[Alignment2D] = partial_related_search(pc_list, eval_policy, target_idx)
    method = "{}_search".format(policy_name)
    save_related_eval_answer(answers, dataset, method)


def show(dataset, metric):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    problems = [p for p in problems if p.score >= 0.5]
    problems = problems[:10]
    dataset = dataset + "_ten"
    answers = load_related_eval_answer(dataset, "{}_search".format(metric))
    tokenizer = get_tokenizer()
    target_seg_idx = 0
    for p, a in join_p_withother(problems, answers):
        print(rei_to_text(tokenizer, p))
        for seg_idx in range(p.seg_instance.text2.get_seg_len()):
            score = a.contribution.table[target_seg_idx][seg_idx]
            tokens_ids = p.seg_instance.text2.get_tokens_for_seg(seg_idx)
            text = ids_to_text(tokenizer, tokens_ids)
            print("{0:.2f}: {1}".format(score, text))


def show2():
    dataset = "dev_sm"
    score_path = os.path.join(output_path, "qtype", "related_scores", "MMDE_dev_mmd_Z.score")
    raw_json = json.load(open(score_path, "r"))
    answers = parse_related_eval_answer_from_json(raw_json)
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    tokenizer = get_tokenizer()

    for p, a in join_p_withother(problems, answers):
        print(rei_to_text(tokenizer, p))
        for target_seg_idx in [0, 1]:
            print("Target seg_idx={}".format(target_seg_idx))
            for seg_idx in range(p.seg_instance.text2.get_seg_len()):
                score = a.contribution.table[target_seg_idx][seg_idx]
                tokens_ids = p.seg_instance.text2.get_tokens_for_seg(seg_idx)
                text = ids_to_text(tokenizer, tokens_ids)
                print("{0:.2f}: {1}".format(score, text))
        break


def main():
    # run_search("dev", "partial_relevant")
    run_search("dev_word", "erasure_no_seg")
    # show("dev_word", "partial_relevant2")


if __name__ == "__main__":
    show2()
