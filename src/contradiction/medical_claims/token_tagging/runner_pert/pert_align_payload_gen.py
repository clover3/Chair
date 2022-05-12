import os
from typing import List, Iterator

from alignment import RelatedEvalInstance
from alignment.extract_feature import eval_all_perturbations
from bert_api import SegmentedInstance
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText
from bert_api.task_clients.nli_interface.nli_interface import NLIInput, save_nli_inputs_to_jsonl
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, \
    load_alamri_split
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.task_async_helper import NonDuplicateJobEnum, StoreAndIter, JobPayloadSaver
from epath import job_man_dir
from misc_lib import TEL, exist_or_mkdir


def enum_rei_for_alamri(tokenizer, p: AlamriProblem) -> List[RelatedEvalInstance]:
    text1_tokens = p.text1.split()
    text2_tokens = p.text2.split()
    t1: SegmentedText = token_list_to_segmented_text(tokenizer, text1_tokens)
    t2: SegmentedText = token_list_to_segmented_text(tokenizer, text2_tokens)
    rei_list = []
    for q, d in [(t1, t2), (t2, t1)]:
        rei = RelatedEvalInstance("dummy_id", SegmentedInstance(q, d), 0)
        rei_list.append(rei)
    return rei_list


def main():
    split = "dev"
    problems: List[AlamriProblem] = load_alamri_split(split)
    tokenizer = get_tokenizer()
    sai: StoreAndIter[NLIInput] = StoreAndIter([0.])
    dummy_nli_client = NonDuplicateJobEnum(NLIInput.str_hash, sai.forward_fn)

    def enum_unique_rei() -> Iterator[NLIInput]:
        for p in TEL(problems):
            rei_list = enum_rei_for_alamri(tokenizer, p)
            eval_all_perturbations(dummy_nli_client, rei_list)
            yield from sai.pop_items()

    job_name = "alamri_{}_perturbation_payloads".format(split)
    save_dir = os.path.join(job_man_dir, job_name)
    exist_or_mkdir(save_dir)
    def save_fn(job_id, items: List[NLIInput]):
        save_path = os.path.join(save_dir, str(job_id))
        return save_nli_inputs_to_jsonl(save_path, items)

    job_payload_saver = JobPayloadSaver(save_fn, 1000)
    job_payload_saver.run_with_itr(enum_unique_rei())


if __name__ == "__main__":
    main()
