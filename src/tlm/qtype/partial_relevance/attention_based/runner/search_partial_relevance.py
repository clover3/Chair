import os
from typing import List, Iterable, Dict

import numpy as np
import scipy.special

from bert_api.client_lib import BERTClient
from cache import load_pickle_from
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from port_info import MMD_Z_PORT
from tlm.data_gen.doc_encode_common import split_by_window
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.enum_util import enum_samples
from tlm.qtype.partial_relevance.attention_based.search_partial_relevance_common import TwoPieceQueryPart, \
    query_info_to_tuple, Instance, \
    enum_helper


def build_ft_list(query_info_dict: Dict[str, QueryInfo]):
    seen = set()
    ft_list = []
    for k, info in query_info_dict.items():
        sig = query_info_to_tuple(info)
        if sig not in seen:
            ft_list.append(TwoPieceQueryPart.from_tuple(sig))
        seen.add(sig)
    return ft_list


def join_ft_ct(ft: TwoPieceQueryPart, ct):
    items = [ft.head, ct, ft.tail]
    items = [item for item in items if item]
    return " ".join(items)


def do_search_job(client: BERTClient, ft_list: List[TwoPieceQueryPart], instance_itr: Iterable[Instance]):
    tokenizer = get_tokenizer()

    def to_token_ids(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    max_try = 50 * 1000
    batch_size = 100

    skip = 4
    for idx, inst in enumerate(instance_itr):
        if idx < skip:
            continue
        seg2 = inst.doc_tokens_ids
        if inst.logits > 0.5:
            continue
        # find ft which makes decision true
        print("{} : {}".format(idx, inst.get_query_rep()))
        # print("Logits: {0:.2f}".format(inst.logits))

        doc_text = pretty_tokens(tokenizer.convert_ids_to_tokens(seg2), True)
        # print("Doc: ")
        # print(doc_text)
        n_seen = 0
        for ft_batch in split_by_window(ft_list, batch_size):
            payload = []
            for ft in ft_batch:
                new_query = join_ft_ct(ft, inst.ct)
                seg1 = to_token_ids(new_query)
                payload.append((seg1, seg2))
            logits = client.request_multiple_from_ids(payload)
            probs = scipy.special.softmax(logits, axis=-1)[:, 1]
            is_rel = np.less(0.5, probs)
            if np.count_nonzero(is_rel):
                print()
                rel_list = [i for i in range(len(probs)) if is_rel[i]]
                print(" / ".join([str(ft_batch[i]) for i in rel_list]))
                break
            else:
                n_seen += batch_size
                print("\r{} checked".format(n_seen), end="")

                if n_seen >= max_try:
                    break


def main():
    run_name = "qtype_2Y_v_train_120000"
    save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    _, query_info_dict = load_pickle_from(os.path.join(save_dir, "0"))
    itr = enum_samples(save_dir)
    itr = enum_helper(itr, query_info_dict)
    ft_list = build_ft_list(query_info_dict)
    print("{} ft in list".format(len(ft_list)))
    client = BERTClient("http://localhost", MMD_Z_PORT, 512)
    do_search_job(client, ft_list, itr)


if __name__ == "__main__":
    main()
