import os
import pickle
import random
from typing import List, Dict, Tuple

from cache import load_from_pickle
from cpath import qtype_root_dir
from dataset_specific.msmarco.common import QueryID
from tlm.qtype.content_functional_parsing.qid_to_content_tokens \
    import load_query_info_dict, QueryInfo, structured_qtype_text


def gen_A(split):
    print("Loading query info")
    qtype_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    # These are from training split
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    functional_tokens_list: List[str] = list(qtype_id_mapping.keys())
    train_qtype_info_dict: Dict[str, QueryInfo] = load_query_info_dict("train")
    functional_token_head_tail_d: Dict[str, Tuple[str, str]] = structured_qtype_text(train_qtype_info_dict)
    print("Now generating queries.")
    new_query_list: List[QueryInfo] = []
    num_gen_per_query = 10
    for qid, info in qtype_info_dict.items():
        sampled_func_tokens: List[str] = random.sample(functional_tokens_list, num_gen_per_query)
        for j in range(num_gen_per_query):
            new_query_id = QueryID("{}_{}".format(qid, j))
            func_tokens: str = sampled_func_tokens[j]
            func_tokens_list = func_tokens.split()
            head, tail = functional_token_head_tail_d[func_tokens]
            all_tokens = head.split() + info.content_span.split() + tail.split()
            out_s_list = head.split() + ["["] + info.content_span.split() + ["]"] + tail.split()
            new_query = " ".join(all_tokens)
            new_query_info = QueryInfo(
                new_query_id,
                new_query,
                info.content_span,
                func_tokens_list,
                out_s_list
            )
            new_query_list.append(new_query_info)
    return new_query_list


def main():
    for split in ["train", "dev", "test"]:
        print(split)
        new_query_list = gen_A(split)
        save_name = "NewQuerySetA{}".format(split)
        pickle.dump(new_query_list, open(os.path.join(qtype_root_dir, save_name), "wb"))
        break


if __name__ == "__main__":
    main()