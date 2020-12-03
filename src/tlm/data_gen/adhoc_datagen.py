import collections
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple

from data_generator.data_parser.robust import load_robust04_query
from data_generator.data_parser.robust2 import load_qrel, load_bm25_best
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import enum_passage
from tf_util.record_writer_wrap import RecordWriterWrap, write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.classification_common import ClassificationInstance, encode_classification_instance
from tlm.data_gen.pairwise_common import generate_pairwise_combinations, write_pairwise_record
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict


class EncoderInterface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def encode(self, query, tokens):
        # returns tokens, segmend_ids
        pass


class FirstSegmentAsDoc(EncoderInterface):
    def __init__(self, max_seq_length):
        super(FirstSegmentAsDoc, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens):
        content_len = self.max_seq_length - 3 - len(query_tokens)
        second_tokens = tokens[:content_len]
        out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
        return [(out_tokens, segment_ids)]


class MultiWindow(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindow, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        self.all_segment_as_doc = AllSegmentAsDoc(src_window_size)

    def encode(self, query_tokens, tokens):
        insts = self.all_segment_as_doc.encode(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        for tokens, segment_ids in insts:
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)

            if len(out_tokens) > self.max_seq_length:
                break

        return [(out_tokens[:self.max_seq_length], out_segment_ids[:self.max_seq_length])]


class AllSegmentAsDoc(EncoderInterface):
    def __init__(self, max_seq_length):
        super(AllSegmentAsDoc, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for second_tokens in enum_passage(tokens, content_len):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class RobustPairwiseTrainGen:
    def __init__(self, encoder, max_seq_length):
        self.data = load_robust_tokens_for_train()
        assert len(self.data) == 174787
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    def generate(self, query_list):
        all_insts = []
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)
            pos_inst = []
            neg_inst = []
            for doc_id, score in judgement.items():
                tokens = self.data[doc_id]
                insts = self.encoder.encode(query_tokens, tokens)
                if score > 0:
                    pos_inst.extend(insts)
                else:
                    neg_inst.extend(insts)
            inst_per_query = generate_pairwise_combinations(neg_inst, pos_inst)

            all_insts.extend(inst_per_query)

        return all_insts

    def write(self, insts, out_path):
        write_pairwise_record(self.tokenizer, self.max_seq_length, insts, out_path)


class RobustPredictGen:
    def __init__(self, encoder, max_seq_length, top_k=100):
        self.data = self.load_tokens_from_pickles()
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.galago_rank = load_bm25_best()
        self.top_k = top_k

        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    @staticmethod
    def load_tokens_from_pickles():
        path = os.path.join(sydney_working_dir, "RobustPredictTokens3", "1")
        return pickle.load(open(path, "rb"))

    def generate(self, query_list):
        all_insts = []
        for query_id in query_list:
            if query_id not in self.galago_rank:
                continue
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)
            for doc_id, _, _ in self.galago_rank[query_id][:self.top_k]:
                tokens = self.data[doc_id]
                insts = self.encoder.encode(query_tokens, tokens)
                all_insts.extend([(t, s, doc_id) for t, s in insts])
        return all_insts

    def write(self, insts, out_path):
        writer = RecordWriterWrap(out_path)
        f = open(out_path+".info", "wb")
        doc_id_list = []
        for inst in insts:
            tokens, segment_ids, doc_id = inst
            feature = get_basic_input_feature(self.tokenizer, self.max_seq_length, tokens, segment_ids)
            doc_id_list.append(doc_id)

            writer.write_feature(feature)

        pickle.dump(doc_id_list, f)
        writer.close()


class SeroRobustPairwiseTrainGen(RobustPairwiseTrainGen):
    def __init__(self, max_seq_length):
        super(SeroRobustPairwiseTrainGen, self).__init__(SeroRobustPairwiseTrainGen, max_seq_length)
        self.max_query_len = 20

    def write(self, insts, out_path):
        writer = RecordWriterWrap(out_path)

        def tokens_to_int_feature(tokens):
            return create_int_feature(self.tokenizer.convert_tokens_to_ids(tokens))

        for inst in insts:
            query_tokens, content_tokens, label = inst
            feature = collections.OrderedDict()
            feature['query'] = tokens_to_int_feature(query_tokens[:self.max_query_len])
            feature['content'] = tokens_to_int_feature(content_tokens[:self.max_seq_length])
            feature['label_ids'] = create_int_feature([label])
            writer.write_feature(feature)
        writer.close()


class RobustPointwiseTrainGen:
    def __init__(self, encoder, max_seq_length):
        self.data = load_robust_tokens_for_train()
        assert len(self.data) == 174787
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    def generate(self, query_list) -> List[ClassificationInstance]:
        all_insts = []
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)
            for doc_id, score in judgement.items():
                tokens = self.data[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(query_tokens, tokens)
                label = 1 if score > 0 else 0

                for tokens_seg, seg_ids in insts:
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    all_insts.append(ClassificationInstance(tokens_seg, seg_ids, label))

        return all_insts

    def write(self, insts: List[ClassificationInstance], out_path: str):
        def encode_fn(inst: ClassificationInstance) -> collections.OrderedDict :
            return encode_classification_instance(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))


class RobustPointwiseTrainGenEx:
    def __init__(self, encoder, max_seq_length):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()


    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d


    def generate(self, query_list) -> List[ClassificationInstance]:
        neg_k = 1000
        all_insts = []
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)

            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            print("Total of {} docs".format(len(target_docs)))

            for doc_id in target_docs:
                tokens = self.data[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(query_tokens, tokens)
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0

                for tokens_seg, seg_ids in insts:
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    all_insts.append(ClassificationInstance(tokens_seg, seg_ids, label))

        return all_insts

    def write(self, insts: List[ClassificationInstance], out_path: str):
        def encode_fn(inst: ClassificationInstance) -> collections.OrderedDict :
            return encode_classification_instance(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))

