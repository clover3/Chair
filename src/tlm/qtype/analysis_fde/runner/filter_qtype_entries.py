import os
import pickle
from typing import List, Dict

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from cpath import output_path
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import parse_q_weight_inner
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import load_query_info_dict, QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


def parse_q_weight_output(raw_prediction_path, data_info) -> List[QTypeInstance]:
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    step = 10
    for idx, e in enumerate(viewer):
        if idx % step == 0:
            yield parse_q_weight_inner(data_info, e)


def load_parse(info, raw_prediction_path, split):
    print("Parsing predictions...")
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    print("Reading QType Entries")
    qtype_entries: List[QTypeInstance] = list(parse_q_weight_output(raw_prediction_path, info))
    return qtype_entries, query_info_dict


def run_parse_sample():
    # MMD_train_qe_de_distill_base_prob
    run_name = "qtype_2X_v_train_200000"
    sample_save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    exist_or_mkdir(sample_save_dir)
    split = "train"
    info_path = os.path.join(job_man_dir, "MMD_train_qe_de_distill_base_prob_info2")
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_path, f_handler.get_mapping(), f_handler.drop_kdp())

    for job_id in range(14, 37):
        print(job_id)
        try:
            pred_path = os.path.join(output_path, "qtype", run_name, str(job_id))
            if not os.path.exists(pred_path):
                print(pred_path + "NOT FOUND")
                continue
            qtype_entries, query_info_dict = load_parse(info, pred_path, split)
            obj = qtype_entries, query_info_dict
            pickle_path = os.path.join(sample_save_dir, str(job_id))
            pickle.dump(obj, open(pickle_path, "wb"))
        except pickle.UnpicklingError as e:
            print(e)




def main():
    run_parse_sample()


if __name__ == "__main__":
    main()