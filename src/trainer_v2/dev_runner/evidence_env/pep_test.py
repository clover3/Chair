import os
import pickle
from typing import List

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from data_generator2.segmented_enc.es_nli.path_helper import get_evidence_selected0_path
from port_info import LOCAL_DECISION_PORT
from tlm.data_gen.base import combine_with_sep_cls
from trainer_v2.evidence_selector.enviroment import PEPClient


def main():
    split = "dev"
    output_dir = at_output_dir("tfrecord", "nli_pep1")
    source_path = get_evidence_selected0_path(split)
    src_data: List[PHSegmentedPair] = pickle.load(open(source_path, "rb"))
    tokenizer = get_tokenizer()

    segment_len = 300

    pep_payload = []
    info = []
    for e in src_data[:3]:
        for i in [0, 1]:
            tokens, segment_ids = combine_with_sep_cls(segment_len, e.get_partial_prem(i), e.get_partial_hypo(i))
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            while len(input_ids) < segment_len:
                input_ids.append(0)
                segment_ids.append(0)
            state = (input_ids, segment_ids)

            action = [1 for _ in input_ids]
            pep_payload.append((action, state))
            info.append((tokens, e.nli_pair.get_label_as_int()))

    client = PEPClient("localhost", LOCAL_DECISION_PORT)
    output = client.request(pep_payload)

    for out, info in zip(output, info):
        print(out)
        print(info)


if __name__ == "__main__":
    main()