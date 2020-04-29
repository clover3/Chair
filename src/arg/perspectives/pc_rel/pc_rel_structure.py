from collections import OrderedDict
from typing import List
from typing import Tuple, Dict

from arg.perspectives.select_paragraph_perspective import ParagraphClaimPersFeature
from data_generator.tokenizer_wo_tf import FullTokenizer
from misc_lib import DataIDGen
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


# For each of claim or persepctive (not combination)
# This encoding format is to match pretrained MSMarco


def to_retrieval_format(tokenizer: FullTokenizer,
                        max_seq_length: int,
                        data_id_gen: DataIDGen,
                        f: ParagraphClaimPersFeature,
                        ) -> Tuple[Dict, List[OrderedDict]]:

    info_list = {}

    def get_feature(tokens1, tokens2, info):
        data_id = data_id_gen.new_id()
        info_list[data_id] = info
        tokens = tokens1 + tokens2
        segment_ids = [0] * len(tokens1) + [1] * len(tokens2)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer,
                                           max_seq_length,
                                           tokens,
                                           segment_ids)
        features['label_ids'] = create_int_feature([0])
        features['data_id'] = create_int_feature([data_id])
        return features

    ordered_dict_list = []
    for scored_paragraph in f.feature:
        tokens2 = scored_paragraph.paragraph.subword_tokens
        claim_tokens = tokenizer.tokenize(f.claim_pers.claim_text)
        p_tokens = tokenizer.tokenize(f.claim_pers.p_text)
        data_info_c = {
            'cid': f.claim_pers.cid,
        }
        out_f = get_feature(claim_tokens, tokens2, data_info_c)
        ordered_dict_list.append(out_f)

        data_info_p = {
            'pid': f.claim_pers.pid
        }
        out_f = get_feature(p_tokens, tokens2, data_info_p)
        ordered_dict_list.append(out_f)

    return info_list, ordered_dict_list
