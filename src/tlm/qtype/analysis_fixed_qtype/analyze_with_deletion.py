import os
import os
import pickle
from typing import List, Dict

import numpy as np

from cache import load_from_pickle, save_to_pickle
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from explain.tf2.deletion_scorer import TokenExEntry
from explain.tf2.deletion_scorer import summarize_deletion_score
from misc_lib import exist_or_mkdir
from scipy_aux import logit_to_score_softmax
from tlm.qtype.analysis_fixed_qtype.parse_qtype_vector import QDistClient
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict
from tlm.qtype.qe_de_res_parse import load_info_jsons
from visualize.html_visual import get_tooltip_cell, Cell, HtmlVisualizer


def check_files():
    dir_path = os.path.join(output_path, "qtype_2J_95000_sensitivity")
    out_dir_path = os.path.join(dir_path, "rename")
    exist_or_mkdir(out_dir_path)
    deletion_offset_list, deletion_per_job = get_offset_list()
    rename_map = {"de_input_ids": "input_ids"}
    old_data_ids = None
    for shift in deletion_offset_list:
        # print(shift)
        load_path = os.path.join(out_dir_path, str(shift))
        data = pickle.load(open(load_path, "rb"))
        data_id_list = []
        for batch in data:
            data_id_list.append(batch["data_id"])
            print(batch['data_id'].shape, batch['logits'].shape)
        data_ids = np.concatenate(data_id_list)
        old_data_ids = data_ids


def transform_results():
    dir_path = os.path.join(output_path, "qtype_2J_95000_sensitivity")
    out_dir_path = os.path.join(dir_path, "rename")
    exist_or_mkdir(out_dir_path)
    deletion_offset_list, deletion_per_job = get_offset_list()
    rename_map = {"de_input_ids": "input_ids"}
    for shift in deletion_offset_list:
        load_path = os.path.join(dir_path, str(shift))
        data = pickle.load(open(load_path, "rb"))
        new_data = []
        for batch in data[:-1]:
            new_batch = {}
            for key in batch:
                if key not in rename_map:
                    new_batch[key] = batch[key]
                else:
                    new_name = rename_map[key]
                    assert new_name not in new_batch
                    new_batch[new_name] = batch[key]

                logits = batch["logits"]
                d_bias = batch["d_bias"]
                logits = logits + np.expand_dims(d_bias, 1)
                new_batch["logits"] = logits
            new_data.append(new_batch)
        save_path = os.path.join(out_dir_path, str(shift))
        pickle.dump(new_data, open(save_path, "wb"))


def print_to_text(summarized_result: List[TokenExEntry], out_file_name):
    tokenizer = get_tokenizer()
    for e in summarized_result:
        tokens = tokenizer.convert_ids_to_tokens(e.input_ids)
        print(" ".join(tokens))
        for loc, vector in e.case_logits_d.items():
            contribution = np.array(e.base_logits) - np.array(vector)
            rank = np.argsort(contribution)[::-1]
            print(loc, tokens[loc])
            for i in range(5):
                type_idx = rank[i]
                print("{0} {1:.2f}".format(type_idx, contribution[type_idx]))
        print()


def print_to_html(data: List[TokenExEntry], info: Dict, qtype_id_to_text, save_name):
    tokenizer = get_tokenizer()
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict("dev")

    base_vectors = np.stack([e.base_logits for e in data], axis=0)
    mean_vector = np.mean(base_vectors, axis=0)
    var_vector = np.mean((base_vectors - mean_vector) ** 2, axis=0)
    var_vector = np.maximum(var_vector, np.ones_like(var_vector) * 1)
    var_vector = np.ones_like(var_vector)
    print(var_vector)
    q_dist_client = QDistClient()
    def transform(vector):
        return vector
        # return (vector - mean_vector) / (var_vector + 1e-8)

    q_dist_cache = load_from_pickle("q_dist_cache")
    # q_dist_cache = {}
    def q_dist_query(text):
        if text in q_dist_cache:
            return q_dist_cache[text]
        r = q_dist_client.query(text)
        q_dist_cache[text] = r
        return r

    html = HtmlVisualizer(save_name, use_tooltip=True)
    for entry in data:
        tokens = tokenizer.convert_ids_to_tokens(entry.input_ids)
        info_e = info[str(entry.data_id)]
        qid = info_e["query"][0]
        q_info = query_info_dict[qid]
        q_dist = q_dist_query(q_info.content_span)
        def transform(vector):
            return vector * np.less(1e-4, q_dist)

        raw_base_logit = np.array(entry.base_logits)
        base_logit = transform(raw_base_logit)
        cells = []
        for idx in range(len(tokens)):
            if tokens[idx] == "[PAD]":
                break
            if tokens[idx] == '[SEP]':
                continue

            if idx in entry.contribution:
                case_logit = transform(np.array(entry.case_logits_d[idx]))
                msg = qtype_vector_with_diff(
                    base_logit,
                    case_logit,
                    qtype_id_to_text)
                cell = get_tooltip_cell(tokens[idx], msg)
                if msg:
                    cell.highlight_score = 80
            else:
                cell = Cell(tokens[idx])
            cells.append(cell)

        msg = qtype_vector_summary(base_logit, qtype_id_to_text)
        passage = pretty_tokens(tokens, True)
        html.write_paragraph(f"Content span: {q_info.content_span}")
        html.write_paragraph(f"Original query: {q_info.query}")
        html.write_paragraph(passage)
        html.write_paragraph(msg)
        html.multirow_print(cells)

    save_to_pickle(q_dist_cache, "q_dist_cache")

def qtype_vector_summary(qtype_vector, qtype_id_to_text):
    rank = np.argsort(qtype_vector)[::-1]
    msg = ""
    for type_idx in rank[:15]:
        func_str = qtype_id_to_text[type_idx]
        if qtype_vector[type_idx] > 1:
            msg += "{0} {1} {2:.2f} <br>".format(type_idx, func_str, qtype_vector[type_idx])
    return msg


def qtype_vector_with_diff(q_vector_base, q_vector_after, qtype_id_to_text):
    diff_vector = q_vector_base - q_vector_after
    f_large_at_begin = np.less(1, q_vector_base)
    f_changed_a_lot = np.less(1, diff_vector)
    mask = np.logical_and(f_changed_a_lot, f_large_at_begin)
    meaningful_diff = diff_vector * mask
    type_id_sorted = np.argsort(diff_vector)[::-1]
    lines = []
    for type_idx in type_id_sorted[:5]:
        if meaningful_diff[type_idx] > 0:
            func_str = qtype_id_to_text[type_idx]
            line = "{0} {1} {2:.2f}({3:.2f})".format(type_idx, func_str,
                                                          q_vector_base[type_idx],
                                                          diff_vector[type_idx]
                                                          )
            lines.append(line)

    msg = ""
    if lines:
        head = "Type_ID Question Base Change <br>"
        msg = head + " <br> ".join(lines)
    return msg



def main():
    dir_path = os.path.join(output_path, "qtype_2J_95000_sensitivity", "rename")
    # save_name = sys.argv[2]
    # info_path = sys.argv[3]
    # info = load_info_jsons(info_path)
    deletion_offset_list, deletion_per_job = get_offset_list()
    summarized_result = summarize_deletion_score(dir_path,
                                                 deletion_per_job,
                                                 32,
                                                 deletion_offset_list,
                                                 logit_to_score_softmax,
                                                 )
    out_file_name = "{}.html".format("qtype_2J_sensitivity")
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    qtype_id_to_text: Dict[int, str] = {v: k for k, v in qtype_id_mapping.items()}
    qtype_id_to_text[0] = "NONE"
    info_path = os.path.join(output_path, "MMD_dev_fixed_qtype_info")
    info = load_info_jsons(info_path)
    print_to_html(summarized_result, info,  qtype_id_to_text, out_file_name)


def get_offset_list():
    st = 0
    deletion_per_job = 20
    num_jobs = 20
    max_offset = st + num_jobs * deletion_per_job
    deletion_offset_list = list(range(st, max_offset, deletion_per_job))
    print(deletion_offset_list)
    return deletion_offset_list, deletion_per_job


if __name__ == "__main__":
    # check_files()
    transform_results()
    main()
