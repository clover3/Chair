import os
from typing import List, Iterable

import numpy as np
import scipy.special

from bert_api.client_lib import BERTClient
from cache import load_pickle_from
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from port_info import MMD_Z_PORT
from tlm.qtype.analysis_fde.fde_module import get_fde_module, FDEModule
from tlm.qtype.enum_util import enum_samples
from tlm.qtype.partial_relevance.attention_based.runner.search_partial_relevance import build_ft_list
from tlm.qtype.partial_relevance.attention_based.search_partial_relevance_common import TwoPieceQueryPart, Instance, \
    enum_helper


def do_search_job(client: BERTClient,
                  fde: FDEModule,
                  ft_list: List[TwoPieceQueryPart],
                  instance_itr: Iterable[Instance]):
    tokenizer = get_tokenizer()

    def to_token_ids(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    skip = 4
    for idx, inst in enumerate(instance_itr):
        seg2 = inst.doc_tokens_ids
        if inst.logits > 0.5:
            continue
        # find ft which makes decision true
        print("Original Query: ", inst.get_query_rep())
        print("Logits: {0:.2f}".format(inst.logits))

        doc_text = pretty_tokens(tokenizer.convert_ids_to_tokens(seg2), True)
        print("Doc: ")
        print(doc_text)
        found_ft = fde.get_promising(inst.ct, doc_text)
        if found_ft:
            payload = []
            print("{} ft founds by fde".format(len(found_ft)))
            found_ft = found_ft[:10]
            for ft_rep in found_ft:
                new_query = ft_rep.replace("[MASK]", inst.ct)
                seg1 = to_token_ids(new_query)
                payload.append((seg1, seg2))
            logits = client.request_multiple_from_ids(payload)
            probs = scipy.special.softmax(logits, axis=-1)[:, 1]
            is_rel = np.less(0.5, probs)
            if np.count_nonzero(is_rel):
                print()
                rel_list = [i for i in range(len(probs)) if is_rel[i]]
                rel_list_to_print = rel_list[:4]
                print("Print {} of {}".format(len(rel_list_to_print), len(rel_list)))
                print(" / ".join([str(found_ft[i]) for i in rel_list_to_print]))

        if idx > 100:
            break

def main():
    run_name = "qtype_2Y_v_train_120000"
    save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    _, query_info_dict = load_pickle_from(os.path.join(save_dir, "0"))
    itr = enum_samples(save_dir)
    itr = enum_helper(itr, query_info_dict)
    ft_list = build_ft_list(query_info_dict)
    print("{} ft in list".format(len(ft_list)))
    fde = get_fde_module()

    client = BERTClient("http://localhost", MMD_Z_PORT, 512)
    do_search_job(client, fde, ft_list, itr)


if __name__ == "__main__":
    main()
