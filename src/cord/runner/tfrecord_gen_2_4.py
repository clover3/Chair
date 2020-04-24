import os
from typing import List, Dict, Tuple, Callable

from cache import save_to_pickle
from cord.csv_to_galago_indexable import read_csv_as_dict, str_cord_uid, str_title, str_abstract
from cord.data_loader import load_queries
from cord.path_info import cord_working_dir, meta_data_path
from cpath import pjoin
from data_generator.common import get_tokenizer
from data_generator.create_feature import create_int_feature
from data_generator.subword_translate import Subword
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature


def encode_query_and_text(tokenizer, query_str, text, max_seq_length) -> Tuple[List[Subword], List[int]]:
    query_tokens = tokenizer.tokenize(query_str)
    text_tokens = tokenizer.tokenize(text)
    max_avail = max_seq_length - 3 - len(query_tokens)
    text_tokens = text_tokens[:max_avail]
    out_tokens: List[Subword] = ["[CLS]"] + query_tokens + ["[SEP]"] + text_tokens+ ["[SEP]"]
    segment_ids: List[int] = [0] * (len(query_tokens) + 2) + [1] * (len(text_tokens) + 1)
    return out_tokens, segment_ids


def tf_record_gen(ranked_list: Dict[str, List[GalagoDocRankEntry]],
                  queries: Dict,
                  text_reader: Callable[[str], str],
                  output_path,
                  max_seq_length: int,
                  data_info_save_name,
                  ):
    writer = RecordWriterWrap(output_path)
    tokenizer = get_tokenizer()
    dummy_label = 0

    data_id_idx = 0
    data_id_info = {}
    for query_id_str in ranked_list:
        query_rep = queries[query_id_str]
        query_str = query_rep['query']

        for ranked_entry in ranked_list[query_id_str]:
            data_id = data_id_idx
            data_id_idx += 1
            data_id_info[data_id] = (query_id_str, ranked_entry.doc_id)
            text = text_reader(ranked_entry.doc_id)
            tokens, segment_ids = encode_query_and_text(tokenizer, query_str, text, max_seq_length)
            features = get_basic_input_feature(tokenizer,
                                               max_seq_length,
                                               tokens,
                                               segment_ids)
            features['label_ids'] = create_int_feature([dummy_label])
            features['data_id'] = create_int_feature([data_id])
            writer.write_feature(features)

    save_to_pickle(data_id_info, data_info_save_name)
    writer.close()


def main():
    queries = load_queries()
    bm25_path = pjoin(cord_working_dir, "youngwoo_bm25_query")
    ranked_list:  Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(bm25_path)
    out_path = os.path.join(cord_working_dir, "tfrecord_2_4")
    max_seq_length = 512

    meat_data: List[Dict] = read_csv_as_dict(meta_data_path)

    text_dict = {}
    for e in meat_data:
        text_dict[e[str_cord_uid]] = e[str_title] + " " + e[str_abstract]

    def get_text_from_doc_id(doc_id:str) -> str:
        return text_dict[doc_id]

    data_info_save_name = "data_info_save"
    tf_record_gen(ranked_list, queries, get_text_from_doc_id, out_path, max_seq_length, data_info_save_name)


if __name__ == "__main__":
    main()
