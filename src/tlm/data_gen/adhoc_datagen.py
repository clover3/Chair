import collections
import os
import pickle
from abc import ABC, abstractmethod

from data_generator.common import get_tokenizer
from data_generator.data_parser.robust import load_robust04_query
from data_generator.data_parser.robust2 import load_2k_rank, load_qrel
from data_generator.job_runner import sydney_working_dir
from misc_lib import pick1
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature, get_basic_input_feature_as_list
from tlm.data_gen.bert_data_gen import create_int_feature

robust_chunk_num = 32


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

    def encode(self, query_tokens, tokens):
        content_len = self.max_seq_length - 3 - len(query_tokens)
        cursor = 0
        insts = []
        while cursor < len(tokens):
            st = cursor
            ed = cursor + content_len
            second_tokens = tokens[st:ed]
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            cursor += content_len
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class RobustTrainGen:
    def __init__(self, encoder, max_seq_length):
        self.data = self.load_from_pickles()
        assert len(self.data) == 174787
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    @staticmethod
    def load_from_pickles():
        data = {}
        for i in range(robust_chunk_num):
            path = os.path.join(sydney_working_dir, "RobustTokensClean", str(i))
            d = pickle.load(open(path, "rb"))
            if d is not None:
                data.update(d)
        return data

    def generate(self, query_list):
        all_insts = []
        cnt = 0
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
            print("pos_insts", len(pos_inst))
            print("neg_insts", len(neg_inst))

            if len(pos_inst) > len(neg_inst):
                major_inst = pos_inst
                minor_inst = neg_inst
                pos_idx = 0
            else:
                major_inst = neg_inst
                minor_inst = pos_inst
                pos_idx = 1

            for idx, entry in enumerate(major_inst):
                entry2 = pick1(minor_inst)

                pos_entry = [entry, entry2][pos_idx]
                neg_entry = [entry, entry2][1-pos_idx]

                if cnt < 10:
                    cnt += 1
                    print(pos_entry)
                    print(neg_entry)
                all_insts.append((pos_entry, neg_entry))

        return all_insts

    def write(self, insts, out_path):
        writer = RecordWriterWrap(out_path)
        for inst in insts:
            (tokens, segment_ids), (tokens2, segment_ids2) = inst

            input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length,
                                                                                 tokens, segment_ids)
            features = collections.OrderedDict()
            features["input_ids1"] = create_int_feature(input_ids)
            features["input_mask1"] = create_int_feature(input_mask)
            features["segment_ids1"] = create_int_feature(segment_ids)
            input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length,
                                                                                 tokens2, segment_ids2)
            features["input_ids2"] = create_int_feature(input_ids)
            features["input_mask2"] = create_int_feature(input_mask)
            features["segment_ids2"] = create_int_feature(segment_ids)

            writer.write_feature(features)
        writer.close()


class RobustPredictGen:
    def __init__(self, encoder, max_seq_length, top_k=100):
        self.data = self.load_tokens_from_pickles()
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_query()
        self.galago_rank = load_2k_rank()
        self.top_k = top_k

        self.encoder = encoder
        self.tokenizer = get_tokenizer()

    @staticmethod
    def load_tokens_from_pickles():
        path = os.path.join(sydney_working_dir, "RobustPredictTokensClean", "1")
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
                all_insts.extend([(t,s, doc_id) for t, s in insts])
                print(query_id, doc_id)
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


class SeroRobustTrainGen(RobustTrainGen):
    def __init__(self, max_seq_length):
        super(SeroRobustTrainGen, self).__init__(SeroRobustTrainGen, max_seq_length)
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
