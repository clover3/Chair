import os
import pickle

from cache import save_list_to_jsonl
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from tlm.qtype.partial_relevance.problem_builder import build_eval_instances, word_segment_w_indices
from tlm.qtype.partial_relevance.runner.sent_tokenize_dev import sentence_segment_w_indices


def main_small():
    dataset = "dev_sm"
    info_path = os.path.join(job_man_dir, "MMDE_dev_info")
    raw_prediction_path = os.path.join(output_path, "qtype", "MMDE_dev_mmd_Z.score")
    items = build_eval_instances(info_path, raw_prediction_path, sentence_segment_w_indices, 10)
    save_path = os.path.join(output_path, "qtype", "MMDE_{}_problems.json".format(dataset))
    save_list_to_jsonl(items, save_path)


def main():
    info_path = os.path.join(job_man_dir, "MMDE_dev_info")
    raw_prediction_path = os.path.join(output_path, "qtype", "MMDE_dev_mmd_Z.score")
    items = build_eval_instances(info_path, raw_prediction_path, sentence_segment_w_indices, 1000)
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems.json")
    save_list_to_jsonl(items, save_path)


def token_level():
    info_path = os.path.join(job_man_dir, "MMDE_dev_info")
    raw_prediction_path = os.path.join(output_path, "qtype", "MMDE_dev_mmd_Z.score")
    items = build_eval_instances(info_path, raw_prediction_path, word_segment_w_indices, 1000)
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_word_problems.json")
    save_list_to_jsonl(items, save_path)


if __name__ == "__main__":
    # print("Building for small set")
    # main_small()
    # print("Building for larger set")
    # main()
    token_level()
