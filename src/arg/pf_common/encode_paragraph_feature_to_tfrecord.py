from collections import OrderedDict
from typing import List

from arg.pf_common.base import ScoreParagraph, ParagraphFeature
from data_generator.subword_translate import Subword
from data_generator.tokenizer_wo_tf import FullTokenizer
from list_lib import lmap
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def format_paragraph_features(tokenizer: FullTokenizer,
                              max_seq_length: int,
                              para_feature: ParagraphFeature) -> List[OrderedDict]:
    text1 = para_feature.datapoint.text1
    tokens1 = tokenizer.tokenize(text1)
    text2 = para_feature.datapoint.text2
    tokens2 = tokenizer.tokenize(text2)
    label: int = int(para_feature.datapoint.label)

    def encode(score_paragraph: ScoreParagraph) -> OrderedDict:
        para_tokens: List[Subword] = score_paragraph.paragraph.subword_tokens

        tokens = tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"] + para_tokens + ["[SEP]"]
        segment_ids = [0] * (len(tokens1) + 1) + [1] * (len(tokens2) + 1) + [2] * (len(para_tokens)+1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer,
                                           max_seq_length,
                                           tokens,
                                           segment_ids)
        features['label_ids'] = create_int_feature([label])
        return features

    features: List[OrderedDict] = lmap(encode, para_feature.feature)
    return features


