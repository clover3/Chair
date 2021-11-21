import os
import sys
from typing import Dict

from arg.qck.trec_helper import score_d_to_trec_style_predictions
from cpath import output_path
from misc_lib import exist_or_mkdir, tprint
from tlm.qtype.qe_de_res_parse import summarize_score, load_info_jsons, get_pair_id
from tlm.qtype.save_qe_point_scores import parser
from trec.trec_parse import write_trec_ranked_list_entry


def save_to_common_path(pred_file_path: str, info_file_path: str, run_name: str,
                        max_entry: int,
                        score_type: str,
                        shuffle_sort: bool
                        ):
    tprint("Reading info...")
    info: Dict = load_info_jsons(info_file_path)
    score_d = summarize_score(info, pred_file_path, get_pair_id, max, "label_ids")
    ranked_list = score_d_to_trec_style_predictions(score_d, run_name, max_entry, shuffle_sort)
    save_dir = os.path.join(output_path, "ranked_list")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)
    tprint("Saved at : ", save_path)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    save_to_common_path(args.pred_path,
                        args.info_path,
                        args.run_name,
                        int(args.max_entry),
                        args.score_type,
                        args.shuffle_sort
                        )