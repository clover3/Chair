import collections
import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable

import math

from data_generator.data_parser.robust import load_robust04_title_query, load_robust_04_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from evals.parse import load_qrels_structured
from misc_lib import enum_passage, DataIDManager, enum_passage_overlap, average
from tf_util.record_writer_wrap import RecordWriterWrap, write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.classification_common import ClassificationInstance, encode_classification_instance, \
    ClassificationInstanceWDataID, encode_classification_instance_w_data_id
from tlm.data_gen.pairwise_common import generate_pairwise_combinations, write_pairwise_record
from tlm.data_gen.robust_gen.select_supervision.score_selection_methods import *
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens_for_predict

Tokens = List[str]
SegmentIDs = List[int]


class EncoderInterface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def encode(self, query, tokens):
        # returns tokens, segmend_ids
        pass


class EncoderTokenCounterInterface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def count(self, query, tokens) -> int:
        # returns tokens, segmend_ids
        pass


class EncoderTokenCounter2Interface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def count(self, query, tokens) -> List[Tuple[List, List, int]]:
        # returns tokens, segmend_ids
        pass


def get_combined_tokens_segment_ids(query_tokens, second_tokens) -> Tuple[Tokens, SegmentIDs]:
    out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
    return out_tokens, segment_ids


class FirstSegmentAsDoc(EncoderInterface):
    def __init__(self, max_seq_length):
        super(FirstSegmentAsDoc, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens):
        content_len = self.max_seq_length - 3 - len(query_tokens)
        second_tokens = tokens[:content_len]
        out_tokens, segment_ids = get_combined_tokens_segment_ids(query_tokens, second_tokens)
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


class MultiWindowTokenCount(EncoderTokenCounterInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowTokenCount, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        self.all_segment_as_doc = AllSegmentAsDocTokenCounter(src_window_size)

    def count(self, query_tokens, tokens):
        insts = self.all_segment_as_doc.count(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        num_content_tokens_acc = 0
        for tokens, segment_ids, num_content_tokens in insts:
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)
            num_content_tokens_acc += num_content_tokens
            if len(out_tokens) > self.max_seq_length:
                break

        return num_content_tokens_acc



class MultiWindowOverlap(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowOverlap, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        step_size = int(self.window_size / 2)
        self.sub_encoder = OverlappingSegments(src_window_size, step_size)

    def encode(self, query_tokens, tokens):
        insts = self.sub_encoder.encode(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        for idx, (tokens, segment_ids) in enumerate(insts):
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)

            if idx + 1 < len(insts):
                assert len(segment_ids) == self.window_size

            if len(out_tokens) > self.max_seq_length:
                break

        return [(out_tokens[:self.max_seq_length], out_segment_ids[:self.max_seq_length])]



class MultiWindowAllSeg(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowAllSeg, self).__init__(max_seq_length)
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

            if len(out_tokens) == self.max_seq_length:
                yield out_tokens, out_segment_ids
                out_tokens = []
                out_segment_ids = []

            if len(out_tokens) > self.max_seq_length:
                assert False

        if out_tokens:
            yield out_tokens, out_segment_ids


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


class AllSegmentAsDocTokenCounter(EncoderTokenCounter2Interface):
    def __init__(self, max_seq_length):
        super(AllSegmentAsDocTokenCounter, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def count(self, query_tokens, tokens) -> List[Tuple[List, List, int]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for second_tokens in enum_passage(tokens, content_len):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids, len(second_tokens)
            insts.append(entry)
        return insts



class OverlappingSegments(EncoderInterface):
    def __init__(self, max_seq_length, step_size):
        super(OverlappingSegments, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.step_size = step_size

    def encode(self, query_tokens, tokens) -> List[Tuple[Tokens, SegmentIDs]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for second_tokens in enum_passage_overlap(tokens, content_len, self.step_size, True):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class OverlappingSegmentsEx:
    def __init__(self, max_seq_length, step_size):
        self.max_seq_length = max_seq_length
        self.step_size = step_size

    def encode(self, query_tokens, tokens) -> List[Tuple[int, int, List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        raw_entries = []
        cursor = self.step_size - content_len
        while cursor < len(tokens):
            st = cursor if cursor >= 0 else 0
            ed = cursor + content_len
            second_tokens = tokens[st:ed]
            cursor += self.step_size
            e = st, ed, second_tokens
            raw_entries.append(e)

        short_window = content_len - self.step_size
        cursor = self.step_size - short_window
        while cursor < len(tokens):
            st = cursor if cursor >= 0 else 0
            ed = cursor + short_window
            second_tokens = tokens[st:ed]
            cursor += self.step_size
            e = st, ed, second_tokens
            raw_entries.append(e)

        for st, ed, second_tokens in raw_entries:
            out_tokens, segment_ids = self.decorate_tokens(query_tokens, second_tokens)
            entry = st, ed, out_tokens, segment_ids
            insts.append(entry)
        return insts

    def decorate_tokens(self, query_tokens, second_tokens):
        out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
        pad_length = self.max_seq_length - len(out_tokens)
        out_tokens += ["[PAD]"] * pad_length
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1) + [0] * pad_length
        assert len(out_tokens) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        return out_tokens, segment_ids


class PassageSampling(EncoderInterface):
    def __init__(self, max_seq_length):
        super(PassageSampling, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class FirstAndRandom(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(FirstAndRandom, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class GeoSampler(EncoderInterface):
    def __init__(self, max_seq_length, g_factor=0.5):
        super(GeoSampler, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.g_factor = g_factor

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            chance = math.pow(self.g_factor, idx)
            include = random.random() < chance
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class LeadingN(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(LeadingN, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == self.num_segment:
                break

            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class FirstEquiSero(EncoderInterface):
    def __init__(self, max_seq_length, sero_window_size, num_segment):
        super(FirstEquiSero, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment
        self.sero_window_size = sero_window_size

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_per_window = self.sero_window_size - 3 - len(query_tokens)
        sero_content_length = content_per_window * 4
        content_max_len = self.max_seq_length - 3 - len(query_tokens)
        content_len = min(sero_content_length, content_max_len)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
            break
        return insts


class LeadingSegmentsCombined(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(LeadingSegmentsCombined, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment
        self.window_size = int(max_seq_length / num_segment)

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.window_size - 3 - len(query_tokens)
        tokens_extending = []
        segment_ids_extending = []

        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == self.num_segment:
                break
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)

            assert len(tokens_extending) % self.window_size == 0
            assert len(segment_ids_extending) % self.window_size == 0
            tokens_extending.extend(out_tokens)
            segment_ids_extending.extend(segment_ids)
        return [(tokens_extending, segment_ids_extending)]


class RobustPairwiseTrainGen:
    def __init__(self, encoder, max_seq_length):
        self.data = load_robust_tokens_for_train()
        assert len(self.data) == 174787
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_title_query()
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


class RobustPredictGenOld:
    def __init__(self, encoder, max_seq_length, top_k=100, query_type="title"):
        self.data = self.load_tokens_from_pickles()
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
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
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust04_title_query()
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
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list) -> List[ClassificationInstance]:
        neg_k = self.neg_k
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


class RobustTrainGenWDataID:
    def __init__(self, encoder, max_seq_length, query_type="title"):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, data_id_manager: DataIDManager, query_list) -> List[ClassificationInstanceWDataID]:
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

                for passage_idx, (tokens_seg, seg_ids) in enumerate(insts):
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    data_id = data_id_manager.assign({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'passage_idx': passage_idx,
                        'label': label,
                        'tokens': tokens_seg,
                        'seg_ids': seg_ids,
                    })
                    all_insts.append(ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id))

        return all_insts

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
            return encode_classification_instance_w_data_id(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))


class RobustTrainGenSelected:
    def __init__(self, encoder, max_seq_length, scores, query_type="title", target_selection="best"):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.scores: Dict[Tuple[str, str, int], float] = scores
        self.get_target_indices: Callable[[], List[int]] = {
            'best': get_target_indices_get_best,
            'all': get_target_indices_all,
            'first_and_best': get_target_indices_first_and_best,
            'best_or_over_09': get_target_indices_best_or_over_09,
            'random_over_09': get_target_indices_random_over_09
        }[target_selection]

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, data_id_manager: DataIDManager, query_list) -> List[ClassificationInstanceWDataID]:
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
                if label:
                    passage_scores = list([self.scores[query_id, doc_id, idx] for idx, _ in enumerate(insts)])
                    target_indices = self.get_target_indices(passage_scores)
                else:
                    target_indices = [0]
                    n = len(insts)
                    if random.random() < 0.1 and n > 1:
                        idx = random.randint(1, n-1)
                        target_indices.append(idx)

                for passage_idx in target_indices:
                    tokens_seg, seg_ids = insts[passage_idx]
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    data_id = data_id_manager.assign({
                        'doc_id': doc_id,
                        'passage_idx': passage_idx,
                        'label': label,
                        'tokens': tokens_seg,
                        'seg_ids': seg_ids,
                    })
                    all_insts.append(ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id))

        return all_insts

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
            return encode_classification_instance_w_data_id(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))


class RobustTrainGenSelected2:
    def __init__(self, encoder, max_seq_length, query_type,
                 target_selection_fn: Callable[[str, str, List], List[int]]):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()

        self.target_selection_fn: Callable[[str, str, List], List[int]] = target_selection_fn

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list, data_id_manager: DataIDManager) -> List[ClassificationInstanceWDataID]:
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
                target_indices = self.target_selection_fn(query_id, doc_id, insts)

                for passage_idx in target_indices:
                    tokens_seg, seg_ids = insts[passage_idx]
                    assert type(tokens_seg[0]) == str
                    assert type(seg_ids[0]) == int
                    data_id = data_id_manager.assign({
                        'doc_id': doc_id,
                        'passage_idx': passage_idx,
                        'label': label,
                        'tokens': tokens_seg,
                        'seg_ids': seg_ids,
                    })
                    all_insts.append(ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id))

        return all_insts

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
            return encode_classification_instance_w_data_id(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, len(insts))



class RobustTrainGenSelectedVirtualCounter:
    def __init__(self, encoder, max_seq_length, query_type,
                 target_selection_fn: Callable[[str, str, List], List[int]]):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()

        self.target_selection_fn: Callable[[str, str, List], List[int]] = target_selection_fn

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def generate(self, query_list, data_id_manager: DataIDManager) -> List[ClassificationInstanceWDataID]:
        neg_k = 1000
        all_insts = []
        pos_n_segment = []
        neg_n_segment = []
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
                target_indices = self.target_selection_fn(query_id, doc_id, insts)
                n_segment = len(target_indices)
                if label:
                    pos_n_segment.append(n_segment)
                else:
                    neg_n_segment.append(n_segment)

        print("num pos docs: ", len(pos_n_segment))
        print("num neg docs: ", len(neg_n_segment))
        print("avg n_seg per doc [pos]", average(pos_n_segment))
        print("avg n_seg per doc [neg]", average(neg_n_segment))
        return all_insts

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        pass


class RobustTrainTextSize:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def count(self, query_list) -> List[int]:
        neg_k = self.neg_k
        leng_list = []
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
            for doc_id in target_docs:
                tokens = self.data[doc_id]
                n_tokens = self.encoder.count(query_tokens, tokens)
                leng_list.append(n_tokens)

        return leng_list



class RobustTrainWordCount:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000):
        self.data = self.load_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def load_tokens(self):
        tokens_d = load_robust_tokens_for_train()
        tokens_d.update(load_robust_tokens_for_predict(4))
        return tokens_d

    def count(self, query_list) -> List[int]:
        neg_k = self.neg_k
        leng_list = []
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
            for doc_id in target_docs:
                tokens = self.data[doc_id]
                n_tokens = self.encoder.count(query_tokens, tokens)
                leng_list.append(n_tokens)

        return leng_list
