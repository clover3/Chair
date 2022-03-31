import os
from typing import List, Dict

from cpath import at_output_dir
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def filter_ranked_list(source_ranked_list_path, data_id_list, post_fix):
    rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(source_ranked_list_path)

    out_rlg = []
    for qid, items in rlg.items():
        group_no, inner_idx, sent_type, tag_type = qid.split("_")
        data_id = "{}_{}".format(group_no, inner_idx)
        if data_id in data_id_list and "{}_{}".format(sent_type, tag_type) == post_fix:
            out_rlg.extend(items)

    return out_rlg


def get_ranked_list_path(save_name):
    ranked_list_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name)
    return ranked_list_path


def main():
    tag = "mismatch"
    method = "probe"
    save_name = "{}_{}.txt".format(method, tag)
    ranked_list_path = get_ranked_list_path(save_name)
    sent_type = "hypo"
    post_fix = "{}_{}".format(sent_type, tag)
    for label in ["entail", "neutral", "contradiction"]:
        if sent_type == "prem":
            f = open(at_output_dir("token_tagging", "{}_rev_pid.txt".format(label)), "r")
        else:
            f = open(at_output_dir("token_tagging", "{}_pid.txt".format(label)), "r")
        data_id_list = [line.strip() for line in f]
        print(data_id_list)
        print("{} data ids ".format(len(data_id_list)))
        out_rl = filter_ranked_list(ranked_list_path, data_id_list, post_fix)
        save_name = "{}_{}_{}_{}.txt".format(method, label, sent_type, tag)
        out_ranked_list_save_path = get_ranked_list_path(save_name)
        write_trec_ranked_list_entry(out_rl, out_ranked_list_save_path)



if __name__ == "__main__":
    main()