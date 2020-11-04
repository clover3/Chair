import os
import random
from collections import OrderedDict
from typing import Tuple, List

# use top-k candidate as payload
from arg.mpqa.parse import load_for_split, splits, MPQADocSubjectiveInfo
from cpath import output_path
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap, lflatten
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature


def encode(tokenizer, max_seq_length, t: Tuple[str, bool]):
    text, is_correct = t
    tokens1: List[str] = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
    features['label_ids'] = create_int_feature([int(is_correct)])
    return features


def encode_w_data_id(tokenizer, max_seq_length, t: Tuple[str, bool, int]):
    text, is_correct, data_id = t
    tokens1: List[str] = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
    features['label_ids'] = create_int_feature([int(is_correct)])
    features['data_id'] = create_int_feature([int(data_id)])
    return features


def get_inst_from_doc(doc: MPQADocSubjectiveInfo) -> List[Tuple[str, bool]]:
    output = []
    for s in doc.sentences:
        text = doc.get_text(s)
        is_correct = bool(s.subjectivity_annot)
        e = text, is_correct
        output.append(e)
    return output


def make_and_write(split):
    docs = load_for_split(split)
    data: List[Tuple[str, bool]] = lflatten(lmap(get_inst_from_doc, docs))
    max_seq_length = 512
    random.shuffle(data)
    dir_path = os.path.join(output_path, "mpqa")
    tokenizer = get_tokenizer()

    def encode_fn(t: Tuple[str, bool]) -> OrderedDict:
        return encode(tokenizer, max_seq_length, t)
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, split)
    write_records_w_encode_fn(save_path, encode_fn, data)


def main():
    # transform payload to common QCK format
    for split in splits:
        print(split)
        make_and_write(split)


if __name__ == "__main__":
    main()
