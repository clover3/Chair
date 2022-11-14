import json
import os

from bert_api import SegmentedText, SegmentedInstance
from cpath import data_path, output_path
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def main():
    dataset_name = "dev_sent"
    problems = load_mmde_problem(dataset_name)
    tokenizer = get_tokenizer()

    def decode_seg_text(st: SegmentedText):
        return SegmentedText(tokenizer.convert_ids_to_tokens(st.tokens_ids), st.seg_token_indices)

    j_out_list = []
    for p in problems:
        si_decoded = SegmentedInstance(decode_seg_text(p.seg_instance.text1), decode_seg_text(p.seg_instance.text2))
        j = p.to_json()
        j['seg_instance'] = si_decoded.to_json()
        j_out_list.append(j)

    save_path = os.path.join(output_path, "qtype", "{}_decoded.json".format(dataset_name))
    f = open(save_path, "w")
    for j in j_out_list:
        s = json.dumps(j)
        f.write(s + "\n")
    f.close()


if __name__ == "__main__":
    main()
