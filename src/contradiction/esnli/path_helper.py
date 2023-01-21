import os
from typing import List, Iterator

from contradiction.esnli.load_esnli import load_esnli
from contradiction.mnli_ex.nli_ex_common import nli_ex_entry_to_sent_token_label, NLIExEntry, get_nli_ex_entry_qid
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from cpath import output_path, data_path
from data_generator.NLI.enlidef import enli_tags
from misc_lib import path_join


def get_save_path(save_name):
    save_path = os.path.join(output_path, "esnli", "ranked_list", save_name + ".txt")
    return save_path


def get_save_path_ex(split, run_name, tag_type):
    save_name = "{}_{}_{}".format(split, run_name, tag_type)
    save_path = get_save_path(save_name)
    return save_path


def get_binary_save_path_w_opt(run_name, tag_type, metric):
    save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "esnli", "binary_predictions")
    save_path = path_join(dir_save_path, save_name + ".txt")
    return save_path


def get_esnli_trec_style_label_path(label, split):
    save_path = os.path.join(
        data_path, "nli", "esnli",
        "trec_style", "{}_{}.txt".format(label, split))
    return save_path


def load_esnli_binary_label(split, tag_type):
    entries = load_esnli(split, tag_type)
    output: List[SentTokenLabel] = []
    for e in entries:
        output.extend(nli_ex_entry_to_sent_token_label(e, tag_type))
    return output


def nli_ex_entry_to_sent_token_label_v2(e: NLIExEntry) -> Iterator[SentTokenLabel]:
    todo = [
        ("prem", e.p_indices, e.premise),
        ("hypo", e.h_indices, e.hypothesis)
    ]
    for sent_type, indices, text in todo:
        if indices:
            n_tokens = len(text.split())
            binary = [1 if i in indices else 0 for i in range(n_tokens)]
            yield SentTokenLabel(
                get_nli_ex_entry_qid(e, sent_type),
                binary
            )


def load_esnli_binary_label_all(split):
    entries = []
    for tag_type in enli_tags:
        entries.extend(load_esnli(split, tag_type))
    output: List[SentTokenLabel] = []
    for e in entries:
        output.extend(nli_ex_entry_to_sent_token_label_v2(e))
    return output

