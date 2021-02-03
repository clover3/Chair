import os
from typing import List
from typing import NamedTuple

from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap

# tdlt_path = "/mnt/scratch/rahimi/div-exp/data/TDLT/Passage/test_data_filtered/q_rel_passage_content_more_than_100_toks.txt"
# clue_path = "/mnt/scratch/rahimi/div-exp/data/Clueweb/top_50_length_200_600_justext/final_output_manual/query_rel_doc_text_256.txt"


clue_path = os.path.join(data_path, "genex", "clue.txt")
tdlt_path = os.path.join(data_path, "genex", "tdlt.txt")


class TokenizedInstance(NamedTuple):
    word_tokens: List[str]
    subword_tokens: List[str]
    idx_mapping: List[int]


class PackedInstance(NamedTuple):
    word_tokens: List[str]
    subword_tokens: List[str]
    idx_mapping: List[int]

    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]

    @classmethod
    def from_tokenize_instance(cls, tokenizer, max_seq_length, ti: TokenizedInstance):
        sep_idx = ti.subword_tokens.index("[SEP]")

        cut = max_seq_length - 1
        subword_tokens = ti.subword_tokens[:cut]

        a_len = sep_idx + 1
        b_len = len(subword_tokens) - a_len
        pad_len = max_seq_length - len(subword_tokens)
        assert a_len + b_len + pad_len == max_seq_length

        padded_tokens = subword_tokens + ["[PAD]"] * pad_len

        input_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
        input_mask = [1] * (a_len + b_len) + [0] * pad_len
        segments_ids = [0] * a_len + [1] * b_len + [0] * pad_len

        return PackedInstance(ti.word_tokens, ti.subword_tokens, ti.idx_mapping,
                              input_ids, input_mask, segments_ids)


def load_from_path(file_path) -> List[PackedInstance]:
    tokenizer = get_tokenizer()

    def parse_line(line: str) -> TokenizedInstance:
        tokens = line.split()

        idx_mapping = []
        sb_tokens_all = []

        for idx, token in enumerate(tokens):
            sb_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            idx_mapping.extend([idx] * len(sb_tokens))
            sb_tokens_all.extend(sb_tokens)

        return TokenizedInstance(tokens, sb_tokens_all, idx_mapping)

    output: List[TokenizedInstance] = lmap(parse_line, open(file_path, "r"))
    print(len(output), "parsed")
    return list([PackedInstance.from_tokenize_instance(tokenizer, 512, ti) for ti in output])


def load_as_simple_format(file_name):
    data = load_packed(file_name)
    t = list([(e.input_ids, e.input_mask, e.segment_ids) for e in data])
    return t


def load_packed(file_name) -> List[PackedInstance]:
    file_path = {
        'tdlt': tdlt_path,
        'clue': clue_path}[file_name]

    data = load_from_path(file_path)
    return data


def main():
    load_from_path(tdlt_path)


if __name__ == "__main__":
    main()
