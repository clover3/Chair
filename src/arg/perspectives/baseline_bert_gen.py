from collections import OrderedDict
from typing import List

from arg.perspectives.basic_analysis import load_data_point, PerspectiveCandidate
from data_generator.common import get_tokenizer
from data_generator.create_feature import create_int_feature
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature


def baseline_bert_gen(outpath, split):
    tokenizer = get_tokenizer()
    data: List[PerspectiveCandidate] = load_data_point(split)
    max_seq_length = 512

    def enc_to_feature(pc: PerspectiveCandidate) -> OrderedDict:
        seg1 = tokenizer.tokenize(pc.claim_text)
        seg2 = tokenizer.tokenize(pc.p_text)

        input_tokens = ["[CLS]"] + seg1 + ["[SEP]"] + seg2 + ["[SEP]"]
        segment_ids = [0] * (len(seg1) + 2) + [1] * (len(seg2) + 1)

        feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
        feature["label_ids"] = create_int_feature([int(pc.label)])
        return feature

    writer = RecordWriterWrap(outpath)
    for entry in data:
        writer.write_feature(enc_to_feature(entry))
    writer.close()


