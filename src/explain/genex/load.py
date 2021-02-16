import os
from typing import List
from typing import NamedTuple

from cpath import data_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
# tdlt_path = "/mnt/scratch/rahimi/div-exp/data/TDLT/Passage/test_data_filtered/q_rel_passage_content_more_than_100_toks.txt"
# clue_path = "/mnt/scratch/rahimi/div-exp/data/Clueweb/top_50_length_200_600_justext/final_output_manual/query_rel_doc_text_256.txt"
from log_lib import log_variables

clue_path = os.path.join(data_path, "genex", "clue.txt")
tdlt_path = os.path.join(data_path, "genex", "tdlt.txt")
wiki_path = os.path.join(data_path, "genex", "wiki.txt")


class TokenizedInstance(NamedTuple):
    word_tokens: List[str]
    subword_tokens: List[str]
    idx_mapping: List[int]


class QueryDoc(NamedTuple):
    query: List[str]
    doc: List[str]


class PackedInstance(NamedTuple):
    word_tokens: List[str]
    subword_tokens: List[str]
    idx_mapping: List[int]

    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]

    @classmethod
    def from_tokenize_instance_old(cls, tokenizer, max_seq_length, ti: TokenizedInstance):
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

    @classmethod
    def from_tokenize_instance(cls, tokenizer, max_seq_length, ti: TokenizedInstance):
        sep_idx = ti.subword_tokens.index("[SEP]")

        cut = max_seq_length - 1
        subword_tokens = ti.subword_tokens[:cut]

        a_len = sep_idx + 2
        b_len = len(subword_tokens) - sep_idx - 1
        pad_len = max_seq_length - len(subword_tokens) - 1
        assert a_len + b_len + pad_len == max_seq_length

        padded_tokens = ["[CLS]"] + subword_tokens + ["[PAD]"] * pad_len

        input_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
        input_mask = [1] * (a_len + b_len) + [0] * pad_len
        segments_ids = [0] * a_len + [1] * b_len + [0] * pad_len

        idx_mapping = [-1] + ti.idx_mapping
        log_variables(input_ids, input_mask, segments_ids)
        return PackedInstance(ti.word_tokens, ti.subword_tokens, idx_mapping,
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
    return list([PackedInstance.from_tokenize_instance(tokenizer, 512, ti) for ti in output])


def load_as_simple_format(file_name):
    data = load_packed(file_name)
    t = list([(e.input_ids, e.input_mask, e.segment_ids) for e in data])
    return t


def load_packed(file_name) -> List[PackedInstance]:
    file_path = os.path.join(data_path, "genex", "{}.txt".format(file_name))
    data = load_from_path(file_path)
    return data


def load_as_lines(file_name) -> List[str]:
    file_path = os.path.join(data_path, "genex", "{}.txt".format(file_name))
    output = []
    for line in open(file_path, "r"):
        output.append(line)
    return output


def parse_problem(raw_tokens: List[str]) -> QueryDoc:
    sep = "[SEP]"
    idx = raw_tokens.index(sep)
    query = raw_tokens[:idx]
    doc = raw_tokens[idx + 1:]
    return QueryDoc(query, doc)


def load_as_tokens(file_name) -> List[QueryDoc]:
    file_path = os.path.join(data_path, "genex", "{}.txt".format(file_name))
    sep = "[SEP]"
    output = []
    for line in open(file_path, "r"):
        idx = line.find(sep)
        query = line[:idx]
        doc = line[idx+len(sep):]

        q_tokens = query.split()
        doc_tokens = doc.split()
        e = QueryDoc(q_tokens, doc_tokens)
        output.append(e)
    return output


def main():
    load_from_path(tdlt_path)


if __name__ == "__main__":
    main()
