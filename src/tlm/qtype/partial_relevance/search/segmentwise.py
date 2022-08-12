from typing import List

from alignment.data_structure.eval_data_structure import Alignment2D, \
    join_p_withother
from alignment.data_structure.print_helper import rei_to_text
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.get_policy_util import get_eval_policy_wc
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer, \
    load_related_eval_answer
from tlm.qtype.related_search.do_related_search import related_search


def run_search(dataset, policy_name, model_interface="localhost"):
    problems = get_first_ten_positive(dataset)
    target_idx = 0
    dataset = dataset + "_ten"
    eval_policy = get_eval_policy_wc(policy_name, model_interface, target_idx)
    complements = load_complements()
    answers: List[Alignment2D] = related_search(problems, eval_policy, target_idx)
    method = "{}_search2".format(policy_name)
    save_related_eval_answer(answers, dataset, method)


def show(dataset, metric):
    problems = get_first_ten_positive(dataset)
    dataset = dataset + "_ten"
    answers = load_related_eval_answer(dataset, "{}_search2".format(metric))
    tokenizer = get_tokenizer()
    target_seg_idx = 0
    for p, a in join_p_withother(problems, answers):
        print(rei_to_text(tokenizer, p))
        for seg_idx in range(p.seg_instance.text2.get_seg_len()):
            score = a.contribution.table[target_seg_idx][seg_idx]
            tokens_ids = p.seg_instance.text2.get_tokens_for_seg(seg_idx)
            text = ids_to_text(tokenizer, tokens_ids)
            print("{0:.2f}: {1}".format(score, text))


def get_first_ten_positive(dataset):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    problems = [p for p in problems if p.score >= 0.5]
    problems = problems[:10]
    return problems


def main():
    # run_search("dev", "leave_one")
    show("dev", "leave_one")


if __name__ == "__main__":
    main()
