import os
from typing import NamedTuple, List, Tuple

from list_lib import list_equal
from ptorch.ists.chunk_embedding import IndexedChunk, __read_chunks, __read_alignments


class AlignEvalCase(NamedTuple):
    left: List[str]
    right: List[str]
    known_golds: List[Tuple[int, List[int]]]


def load_dataset_from_predefined() -> List[AlignEvalCase]:
    dir_path = "datasets/sts_16/train_2015_10_22.utf-8"
    test_cases = load_dataset_from_dir(dir_path)
    return test_cases


def load_dataset_from_dir(dir_path):
    gold_alignments = os.path.join(dir_path, "STSint.input.images.wa")
    left_chunks_file = os.path.join(dir_path, "STSint.input.images.sent1.chunk.txt")
    right_chunks_file = os.path.join(dir_path, "STSint.input.images.sent2.chunk.txt")
    test_cases = load_dataset(gold_alignments, left_chunks_file, right_chunks_file)
    return test_cases


def load_dataset(gold_alignments, left_chunks_file, right_chunks_file):
    left_chunks: List[List[IndexedChunk]] = __read_chunks(left_chunks_file)
    right_chunks: List[List[IndexedChunk]] = __read_chunks(right_chunks_file)
    overall_alignments = __read_alignments(gold_alignments, left_chunks, right_chunks)

    def parse_chunk_emb_ids(chunk_emb_ids):
        _, _, indices_s = chunk_emb_ids.split("_")
        indices = list(map(int, indices_s.split(",")))
        return indices

    def fill_missing_indices(indices_list, n_tokens):
        indices_list.sort(key=lambda l: l[0])
        cur_idx = 0
        new_indices_list = []
        for indices in indices_list:
            first = indices[0]
            if cur_idx < first:
                new_indices_list.append(list(range(cur_idx, first)))
            new_indices_list.append(indices)
            cur_idx = indices[-1] + 1

        if cur_idx < n_tokens:
            new_indices_list.append(list(range(cur_idx, n_tokens)))
        return new_indices_list

    def translate(sentence_info):
        output_case = {}
        new_indices_list_d = {}
        for left_or_right in ["left", "right"]:
            sentence = sentence_info[f'{left_or_right}_sentence']
            chunk_emb_ids = sentence_info[f'{left_or_right}_chunk_emb_ids']
            tokens = sentence.split()
            indices_list = list(map(parse_chunk_emb_ids, chunk_emb_ids))
            new_indices_list = fill_missing_indices(indices_list, len(tokens))
            new_indices_list_d[left_or_right] = new_indices_list
            n_chunk_sum = sum(map(len, new_indices_list))
            assert n_chunk_sum == len(tokens)

            def form_chunk(indices):
                chunk = " ".join([tokens[i] for i in indices])
                return chunk

            sent_chunks = list(map(form_chunk, new_indices_list))
            output_case[left_or_right] = sent_chunks

        def get_chunk_idx(indices, left_or_right):
            new_indices_list = new_indices_list_d[left_or_right]
            for chunk_idx, new_indices in enumerate(new_indices_list):
                if list_equal(new_indices, indices):
                    return chunk_idx
            raise KeyError

        known_golds = []
        for left_emb_ids, right_emb_ids_set in sentence_info['chunk_emb_mapping'].items():
            left_indices: List[int] = parse_chunk_emb_ids(left_emb_ids)
            left_chunk_idx = get_chunk_idx(left_indices, "left")
            right_chunk_indices = []
            for right_emb_ids in right_emb_ids_set:
                right_indices: List[int] = parse_chunk_emb_ids(right_emb_ids)
                right_chunk_idx = get_chunk_idx(right_indices, "right")
                right_chunk_indices.append(right_chunk_idx)
            known_golds.append((left_chunk_idx, right_chunk_indices))
        return AlignEvalCase(output_case['left'], output_case['right'], known_golds)

    test_cases: List[AlignEvalCase] = [translate(sentence_info) for sid, sentence_info in overall_alignments.items()]
    return test_cases