import collections
import os
import pickle
import random
from typing import List, Tuple, Dict, Callable

from data_generator.data_parser.robust import load_robust_04_query, load_robust04_title_query
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.job_runner import sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, DataIDManager, average
from tf_util.record_writer_wrap import RecordWriterWrap, write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.classification_common import ClassificationInstance, encode_classification_instance, \
    ClassificationInstanceWDataID, encode_classification_instance_w_data_id
from tlm.data_gen.pairwise_common import generate_pairwise_combinations, write_pairwise_record, combine_features
from tlm.data_gen.robust_gen.select_supervision.score_selection_methods import get_target_indices_get_best, \
    get_target_indices_all, get_target_indices_first_and_best, get_target_indices_best_or_over_09, \
    get_target_indices_random_over_09
from tlm.robust.load import load_robust_tokens_for_train, load_robust_tokens, load_robust_tokens_for_predict
from trec.qrel_parse import load_qrels_structured


class RobustPairwiseTrainGen:
    def __init__(self, encoder, max_seq_length, query_type="title"):
        self.data = load_robust_tokens_for_train()
        assert len(self.data) == 174787
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length

        self.queries = load_robust_04_query(query_type)
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


def pos_major_enum(pos_doc_ids, neg_doc_ids):
    for pos_doc_id in pos_doc_ids:
        neg_doc_id = pick1(neg_doc_ids)
        yield pos_doc_id, neg_doc_id


def get_pos_major_repeat_enum(n_repeat):
    def pos_major_repeat_enum(pos_doc_ids, neg_doc_ids):
        for _ in range(n_repeat):
            for pos_doc_id in pos_doc_ids:
                neg_doc_id = pick1(neg_doc_ids)
                yield pos_doc_id, neg_doc_id
    return pos_major_repeat_enum


def neg_major_enum(pos_doc_ids, neg_doc_ids):
    pos_doc_ids_new = list(pos_doc_ids)
    random.shuffle(pos_doc_ids_new)

    pos_doc_cursor = 0

    n_neg_doc = len(neg_doc_ids)
    n_pos_doc = len(pos_doc_ids)
    for neg_doc_id in neg_doc_ids:
        if not n_pos_doc:
            break
        pos_doc_id = pos_doc_ids_new[pos_doc_cursor]
        pos_doc_cursor += 1
        if pos_doc_cursor == n_pos_doc:
            pos_doc_cursor = 0
        yield pos_doc_id, neg_doc_id


def get_neg_major_limit_enum(n_limit):
    def neg_major_enum(pos_doc_ids, neg_doc_ids):
        pos_doc_ids_new = list(pos_doc_ids)
        neg_doc_ids_new = list(neg_doc_ids)
        random.shuffle(pos_doc_ids_new)
        random.shuffle(neg_doc_ids_new)
        pos_doc_cursor = 0
        n_neg_doc = len(neg_doc_ids)
        n_pos_doc = len(pos_doc_ids)
        for neg_doc_id in neg_doc_ids[:n_limit]:
            if not n_pos_doc:
                break
            pos_doc_id = pos_doc_ids_new[pos_doc_cursor]
            pos_doc_cursor += 1
            if pos_doc_cursor == n_pos_doc:
                pos_doc_cursor = 0
            yield pos_doc_id, neg_doc_id
    return neg_major_enum


class RobustPairwiseTrainGen2:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000,
                 enum_pos_neg_method="pos_major_enum",
                 ):
        self.data = load_robust_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k
        if enum_pos_neg_method == "pos_major_enum":
            self.enum_pos_neg_pairs = pos_major_enum
        elif enum_pos_neg_method == "neg_major_enum":
            self.enum_pos_neg_pairs = neg_major_enum
        elif enum_pos_neg_method == "neg_major_limit100":
            self.enum_pos_neg_pairs = get_neg_major_limit_enum(100)
        elif enum_pos_neg_method == "pos_major_repeat_enum":
            self.enum_pos_neg_pairs = get_pos_major_repeat_enum(5)
        elif enum_pos_neg_method == "pos_major_repeat20_enum":
            self.enum_pos_neg_pairs = get_pos_major_repeat_enum(20)
        else:
            print("enum_pos_neg_method {} is not expected".format(enum_pos_neg_method))
            assert False

    def generate(self, query_list):
        neg_k = self.neg_k
        all_insts = []

        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)

            def encode_doc(doc_id):
                tokens = self.data[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(query_tokens, tokens)
                return insts[0]

            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            pos_doc_ids = []
            neg_doc_ids = []

            for doc_id in target_docs:
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                if label:
                    pos_doc_ids.append(doc_id)
                else:
                    neg_doc_ids.append(doc_id)

            for pos_doc_id, neg_doc_id in self.enum_pos_neg_pairs(pos_doc_ids, neg_doc_ids):
                pos_inst: Tuple[List, List] = encode_doc(pos_doc_id)
                tokens_seg1, seg_ids1 = pos_inst
                neg_inst: Tuple[List, List] = encode_doc(neg_doc_id)
                tokens_seg2, seg_ids2 = neg_inst
                inst = (tokens_seg1, seg_ids1), (tokens_seg2, seg_ids2)
                all_insts.append(inst)

        return all_insts

    def write(self, insts, out_path):
        write_pairwise_record(self.tokenizer, self.max_seq_length, insts, out_path)



class RobustPairwiseTrainGenWPosMask:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000,
                 enum_pos_neg_method="pos_major_enum",
                 ):
        self.data = load_robust_tokens()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k
        if enum_pos_neg_method == "pos_major_enum":
            self.enum_pos_neg_pairs = pos_major_enum
        elif enum_pos_neg_method == "neg_major_enum":
            self.enum_pos_neg_pairs = neg_major_enum
        elif enum_pos_neg_method == "neg_major_limit100":
            self.enum_pos_neg_pairs = get_neg_major_limit_enum(100)
        elif enum_pos_neg_method == "pos_major_repeat_enum":
            self.enum_pos_neg_pairs = get_pos_major_repeat_enum(5)
        elif enum_pos_neg_method == "pos_major_repeat20_enum":
            self.enum_pos_neg_pairs = get_pos_major_repeat_enum(20)
        else:
            print("enum_pos_neg_method {} is not expected".format(enum_pos_neg_method))
            assert False

    def generate(self, query_list):
        neg_k = self.neg_k
        all_insts = []

        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]
            query = self.queries[query_id]
            query_tokens = self.tokenizer.tokenize(query)

            def encode_doc(doc_id):
                tokens = self.data[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(query_tokens, tokens)
                return insts[0]

            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            pos_doc_ids = []
            neg_doc_ids = []

            for doc_id in target_docs:
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                if label:
                    pos_doc_ids.append(doc_id)
                else:
                    neg_doc_ids.append(doc_id)

            seen_pos_doc = set()
            for pos_doc_id, neg_doc_id in self.enum_pos_neg_pairs(pos_doc_ids, neg_doc_ids):
                pos_inst: Tuple[List, List] = encode_doc(pos_doc_id)
                tokens_seg1, seg_ids1 = pos_inst
                neg_inst: Tuple[List, List] = encode_doc(neg_doc_id)
                tokens_seg2, seg_ids2 = neg_inst

                use_pos = 1 if pos_doc_id not in seen_pos_doc else 0
                seen_pos_doc.add(pos_doc_id)
                inst = (tokens_seg1, seg_ids1), (tokens_seg2, seg_ids2), use_pos
                all_insts.append(inst)

        return all_insts

    def write(self, insts, out_path):
        writer = RecordWriterWrap(out_path)
        for inst in insts:
            (tokens, segment_ids), (tokens2, segment_ids2), use_pos = inst

            features = combine_features(tokens, segment_ids, tokens2, segment_ids2, self.tokenizer, self.max_seq_length)
            features["use_pos"] = create_int_feature([use_pos])
            writer.write_feature(features)
        writer.close()



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


class RobustPosToNegRate:
    def __init__(self, encoder, max_seq_length, query_type="title", neg_k=1000):
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrels_structured(qrel_path)
        self.max_seq_length = max_seq_length
        self.queries = load_robust_04_query(query_type)
        self.encoder = encoder
        self.galago_rank = load_bm25_best()
        self.neg_k = neg_k

    def generate(self, query_list):
        neg_k = self.neg_k
        n_pos_doc = 0
        n_neg_doc = 0
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            judgement = self.judgement[query_id]

            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]
            target_docs = set(judgement.keys())
            target_docs.update([e.doc_id for e in ranked_list])
            pos_doc_ids = []
            neg_doc_ids = []

            for doc_id in target_docs:
                label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                if label:
                    pos_doc_ids.append(doc_id)
                else:
                    neg_doc_ids.append(doc_id)

            print(len(pos_doc_ids), len(neg_doc_ids))




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