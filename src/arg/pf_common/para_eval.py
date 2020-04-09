from typing import NewType, List, Tuple

from data_generator.subword_translate import Subword

Segment = NewType('Segment', List[Subword])


def split_3segments(input_tokens: Segment) -> Tuple[Segment, Segment, Segment]:
    try:
        idx1 = input_tokens.index("[SEP]")
        idx2 = input_tokens.index("[SEP]", idx1+1)
        idx3 = input_tokens.index("[SEP]", idx2+1)
    except:
        print("Parse fail")
        raise Exception

    return input_tokens[1:idx1], input_tokens[idx1+1:idx2], input_tokens[idx2+1:idx3]


def input_tokens_to_key(input_tokens):
    claim, pers, _ = split_3segments(input_tokens)
    claim_text = " ".join(claim)
    p_text = " ".join(pers)
    key = claim_text + "_" + p_text
    return key