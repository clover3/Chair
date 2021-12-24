import os
from typing import Dict

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from epath import job_man_dir
from tf_util.enum_features import load_record
from tf_util.tfrecord_convertor import take
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict


def load_info():
    split = "train"
    info_path = os.path.join(job_man_dir, "MMD_train_qe_de_distill_base_prob_info2")
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_path, f_handler.get_mapping(), f_handler.drop_kdp())
    return info


def read_info_find_data_id_for_qid():
    qids = ["1012286", "1011953"]
    selected_data_id = {}
    info = load_info()
    for data_id, item in info.items():
        qid = item['query'].query_id
        if qid in qids:
            selected_data_id[data_id] = qid

    print(selected_data_id)


def find_data_id_from_tfrecord():
    tfrecord_path = os.path.join(job_man_dir, "MMD_train_f_de_distill_prob_base", "0")
    data_id_to_qid = {'48235': '1011953', '48236': '1011953', '48237': '1011953', '48238': '1011953', '48239': '1011953', '48240': '1011953', '48241': '1011953', '48242': '1011953', '48243': '1011953', '48244': '1011953', '48245': '1011953', '49521': '1012286', '49522': '1012286', '49523': '1012286', '49524': '1012286', '49525': '1012286', '49526': '1012286', '49527': '1012286', '49528': '1012286', '49529': '1012286', '49530': '1012286', '49531': '1012286'}
    split = "train"
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)

    data_id_to_qid = {int(k): int(v) for k, v in data_id_to_qid.items()}
    for record in load_record(tfrecord_path):
        data_id = take(record["data_id"])[0]
        if data_id in data_id_to_qid:
            qe_input_ids = take(record["q_e_input_ids"])
            corr_qid = data_id_to_qid[data_id]
            q_info = query_info_dict[str(corr_qid)]
            print(" ".join(q_info.out_s_list))
            print(corr_qid, data_id, qe_input_ids)



def find_data_id_from_tfrecord_qe_de():
    tfrecord_path = os.path.join(job_man_dir, "MMD_train_qe_de_distill_base_prob", "0")
    data_id_to_qid = {'48235': '1011953', '48236': '1011953', '48237': '1011953', '48238': '1011953', '48239': '1011953', '48240': '1011953', '48241': '1011953', '48242': '1011953', '48243': '1011953', '48244': '1011953', '48245': '1011953', '49521': '1012286', '49522': '1012286', '49523': '1012286', '49524': '1012286', '49525': '1012286', '49526': '1012286', '49527': '1012286', '49528': '1012286', '49529': '1012286', '49530': '1012286', '49531': '1012286'}
    split = "train"
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)

    data_id_to_qid = {int(k): int(v) for k, v in data_id_to_qid.items()}
    for record in load_record(tfrecord_path):
        data_id = take(record["data_id"])[0]
        if data_id in data_id_to_qid:
            qe_input_ids = take(record["q_e_input_ids"])
            corr_qid = data_id_to_qid[data_id]
            q_info = query_info_dict[str(corr_qid)]
            print(" ".join(q_info.out_s_list))
            print(corr_qid, data_id, qe_input_ids)


if __name__ == "__main__":
    find_data_id_from_tfrecord_qe_de()

