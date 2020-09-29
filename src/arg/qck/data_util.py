from collections import OrderedDict
from typing import List, Iterable, Tuple

from arg.qck.decl import QCInstance, QCKQuery, QCKCandidate
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def encode(tokenizer, max_seq_length, inst: QCInstance) -> OrderedDict:
    tokens1: List[str] = tokenizer.tokenize(inst.query_text)
    max_seg2_len = max_seq_length - 3 - len(tokens1)
    tokens2 = tokenizer.tokenize(inst.candidate_text)
    tokens2 = tokens2[:max_seg2_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features


def write_qc_records(output_path, qc_records):
    data_id_man = DataIDManager()
    instances = collect_info_transform(qc_records, data_id_man)
    tokenizer = get_tokenizer()
    max_seq_length = 512

    def encode_fn(inst: QCInstance):
        return encode(tokenizer, max_seq_length, inst)

    write_records_w_encode_fn(output_path, encode_fn, instances)


def collect_info_transform(data: Iterable[Tuple[QCKQuery, QCKCandidate, bool]], data_id_man: DataIDManager) \
        -> Iterable[QCInstance]:
    for query, candidate, is_correct in data:
        info = {
            'query': query,
            'candidate': candidate
        }
        yield QCInstance(query.text, candidate.text, data_id_man.assign(info), int(is_correct))