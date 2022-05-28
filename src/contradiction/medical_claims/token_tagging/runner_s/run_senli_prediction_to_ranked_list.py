import json
import os
from typing import List, Dict, Tuple

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.solvers.senli_prediction_to_ranked_list import convert_inner
from cpath import output_path
from data_generator.NLI import nli_info
from data_generator.tokenizer_wo_tf import get_tokenizer
from trec.trec_parse import write_trec_ranked_list_entry


def convert(pred_list: List[List[Tuple]],
            info_d: Dict[str, Dict],
            tokenizer):
    run_name = "senli"
    tag_types = ["mismatch", "conflict"]
    ex_logits_joined_d = dict(zip(nli_info.tags, pred_list))
    for tag_type in tag_types:
        save_name = "senli_{}.txt".format(tag_type)
        save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name)
        all_ranked_list = convert_inner(ex_logits_joined_d, info_d, run_name, tag_type, tokenizer)
        write_trec_ranked_list_entry(all_ranked_list, save_path)


# tfrecord/bert_alamri1.pickle
def main():
    data_name = "bert_alamri1"
    tokenizer = get_tokenizer()
    save_dir = os.path.join(output_path, "alamri_annotation1", "tfrecord")
    info_file_path = os.path.join(save_dir, "{}.info".format(data_name))
    run_name = "alamri_bert_nli"
    pred_list = load_from_pickle(run_name)
    info = json.load(open(info_file_path, "r"))
    convert(pred_list, info, tokenizer)


if __name__ == "__main__":
    main()
