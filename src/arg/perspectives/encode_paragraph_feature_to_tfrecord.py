from collections import OrderedDict
from typing import List

from arg.perspectives.select_paragraph import ParagraphClaimPersFeature, ScoreParagraph
from data_generator.subword_translate import Subword
from data_generator.tokenizer_wo_tf import FullTokenizer
from list_lib import lmap
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def format_paragraph_features(tokenizer: FullTokenizer,
                              max_seq_length: int,
                              claim_entry: ParagraphClaimPersFeature) -> List[OrderedDict]:
    claim_text = claim_entry.claim_pers.claim_text
    claim_tokens = tokenizer.tokenize(claim_text)
    p_text = claim_entry.claim_pers.p_text
    p_tokens = tokenizer.tokenize(p_text)
    label: int = int(claim_entry.claim_pers.label)

    def encode(score_paragraph: ScoreParagraph) -> OrderedDict:
        para_tokens: List[Subword] = score_paragraph.paragraph.subword_tokens

        tokens = claim_tokens + ["[SEP]"] + p_tokens + ["[SEP]"] + para_tokens + ["[SEP]"]
        segment_ids = [0] * (len(claim_tokens) + 1) + [1] * (len(p_tokens) + 1) + [2] * (len(para_tokens)+1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer,
                                           max_seq_length,
                                           tokens,
                                           segment_ids)
        features['label_ids'] = create_int_feature([label])
        return features

    features: List[OrderedDict] = lmap(encode, claim_entry.feature)
    return features


