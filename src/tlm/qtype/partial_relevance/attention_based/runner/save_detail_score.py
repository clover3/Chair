import json
import os
from typing import List, Dict

from cpath import at_output_dir
from list_lib import index_by_fn
from misc_lib import exist_or_mkdir
from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalAnswer
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.eval_by_attn import AttentionBrevityDetail
from tlm.qtype.partial_relevance.get_policy_util import get_real_val_eval_policy
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


def get_detail(answer_list: List[RelatedEvalAnswer],
               problem_list: List[RelatedEvalInstance],
               eval_policy: AttentionBrevityDetail,
               ) -> Dict[str, List[Dict]]:
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)

    future_predictions_list = []
    for a in answer_list:
        p: RelatedEvalInstance = pid_to_p[a.problem_id]
        future = eval_policy.get_predictions_for_case(p, a)
        future_predictions_list.append(future)

    eval_policy.inner.do_duty()
    problem_ids: List[str] = [a.problem_id for a in answer_list]

    info_d = {}
    for problem_id, future in zip(problem_ids, future_predictions_list):
        detail_info: List[Dict] = eval_policy.get_detail(future)
        info_d[problem_id] = detail_info
    return info_d


def save_detail(dataset, method, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers = load_related_eval_answer(dataset, method)
    policy = get_real_val_eval_policy("attn_brevity", model_interface, 1)
    detailer = AttentionBrevityDetail(policy)

    info_d = get_detail(answers, problems, detailer)

    save_path = get_attn_detail_save_path(dataset, method)
    json.dump(info_d, open(save_path, "w"))


def get_attn_detail_save_path(dataset, method):
    save_dir = at_output_dir("qtype", "attn_mask_eval_detail")
    exist_or_mkdir(save_dir)
    save_name = "{}_{}.info".format(dataset, method)
    save_path = os.path.join(save_dir, save_name)
    return save_path


def main():
    save_detail("dev_sent", "gradient")


if __name__ == "__main__":
    main()