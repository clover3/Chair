from collections import OrderedDict
from typing import List

import numpy as np

from arg.perspectives.basic_analysis import load_data_point
from arg.perspectives.declaration import PerspectiveCandidate
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer, is_continuation
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_float_feature


def get_word_embedding(emb_model, tokens, dims):
    def join_subwords(sbword_list):
        if len(sbword_list) == 1:
            return sbword_list[0]
        else:
            return "".join(sbword_list[:1] + [s[2:]for s in sbword_list[1:] ])

    cur_word = []
    word_vectors = []

    def append_to_output(cur_word):
        word = join_subwords(cur_word)
        if word in emb_model:
            word_vector = emb_model[word]
        else:
            word_vector = [0] * dims
        for _ in cur_word:
            word_vectors.append(word_vector)

    for subword in tokens:
        if is_continuation(subword):
            cur_word.append(subword)
        else:
            if cur_word:
                append_to_output(cur_word)

            cur_word = [subword]

    if cur_word:
        append_to_output(cur_word)

    assert len(word_vectors) == len(tokens)
    return word_vectors


def gen_with_aux_emb(outpath, aux_embedding_d, split, dims):
    tokenizer = get_tokenizer()
    data: List[PerspectiveCandidate] = load_data_point(split)
    max_seq_length = 512
    zero_vector = [0.] * dims

    not_found = set()
    def get_aux_embedding_fn(cid):
        cid = int(cid)
        if cid in aux_embedding_d:
            return aux_embedding_d[cid]
        else:
            if cid not in not_found:
                not_found.add(cid)
                print("Aux embedding not found", cid)
            return {}

    def enc_to_feature(pc: PerspectiveCandidate) -> OrderedDict:
        emb_model = get_aux_embedding_fn(pc.cid)

        seg1 = tokenizer.tokenize(pc.claim_text)
        seg2 = tokenizer.tokenize(pc.p_text)

        input_tokens = ["[CLS]"] + seg1 + ["[SEP]"] + seg2 + ["[SEP]"]

        aux_emb = get_word_embedding(emb_model, input_tokens, dims)
        aux_emb += (max_seq_length - len(aux_emb)) * [zero_vector]
        aux_emb = np.array(aux_emb)
        flat_aux_emb = np.reshape(aux_emb, [-1])

        segment_ids = [0] * (len(seg1) + 2) + [1] * (len(seg2) + 1)

        feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
        feature["label_ids"] = create_int_feature([int(pc.label)])
        feature["aux_emb"] = create_float_feature(flat_aux_emb)
        return feature

    writer = RecordWriterWrap(outpath)
    for entry in data:
        writer.write_feature(enc_to_feature(entry))
    writer.close()


