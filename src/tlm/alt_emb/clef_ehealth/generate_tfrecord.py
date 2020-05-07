import os
from typing import List, Dict, Tuple

from base_type import FilePath
from cache import save_to_pickle
from cpath import output_path, pjoin
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import load, preload_man
from datastore.table_names import BertTokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import RankedListDict, Query
from list_lib import flatten
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.alt_emb.clef_ehealth.qrel import load_clef_qrels
from tlm.alt_emb.clef_ehealth.split_query import get_query_split
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc
from tlm.data_gen.base import get_basic_input_feature


def write_tfrecord(ranked_list_d: RankedListDict,
                   queries: List[Query],
                   q_rels: Dict[str, List[str]],
                   save_path):
    max_seq_length = 512
    tokenizer = get_tokenizer()
    encoder = AllSegmentAsDoc(max_seq_length)
    writer = RecordWriterWrap(save_path)
    data_id = 0

    data_info = []
    for query in queries:
        if query.qid not in ranked_list_d:
            print("Warning query {} not found".format(query.qid))
            continue
        print(query.qid)
        ranked_list = ranked_list_d[query.qid]
        doc_ids = [doc_entry.doc_id for doc_entry in ranked_list]
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)
        q_tokens = tokenizer.tokenize(query.text)

        for doc_entry in ranked_list:
            try:
                tokens_list: List[List[str]] = load(BertTokenizedCluewebDoc, doc_entry.doc_id)
                tokens = flatten(tokens_list)
                insts: List[Tuple[List, List]] = encoder.encode(q_tokens, tokens)
                for inst in insts:
                    label = doc_entry.doc_id in q_rels[query.qid]

                    input_tokens, segment_ids = inst
                    feature = get_basic_input_feature(tokenizer, max_seq_length, input_tokens, segment_ids)
                    feature["label_ids"] = create_int_feature([int(label)])
                    feature["data_id"] = create_int_feature([int(data_id)])
                    writer.write_feature(feature)

                    data_info.append((data_id, query.qid, doc_entry.doc_id))
                    data_id += 1
            except KeyError as e:
                print("doc {} not found".format(doc_entry.doc_id))

    return data_info


def main():
    train_queries, test_queries = get_query_split()
    out_dir = pjoin(output_path, "eHealth")
    exist_or_mkdir(out_dir)
    train_save_path = pjoin(out_dir, "tfrecord_train")
    test_save_path = pjoin(out_dir, "tfrecord_test")
    ranked_list_path = FilePath(os.path.join(output_path, "eHealth", "bm25_filtered.list"))
    ranked_list: RankedListDict = load_galago_ranked_list(ranked_list_path)
    qrels = load_clef_qrels()

    train_info = write_tfrecord(ranked_list, train_queries, qrels, train_save_path)
    save_to_pickle(train_info, "eHealth_train_info")
    test_info = write_tfrecord(ranked_list, test_queries, qrels, test_save_path)
    save_to_pickle(test_info, "eHealth_test_info")


if __name__ == "__main__":
    main()
