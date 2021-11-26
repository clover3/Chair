from abc import abstractmethod, ABC
from collections import Counter
from typing import List, Dict, Tuple, Iterable

from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, TimeEstimator, TEL
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListI
from tlm.data_gen.qde_common import encode_query_entity_doc_pair_instance, \
    encode_concatenated_query_entity_doc_pair_instance, encode_qde_ids_instance, QueryEntityDocPairInstance, QDE, \
    QDE_as_Ids, QDEPaired, QTypeDE_as_Ids, encode_qtype_de_ids_instance
from tlm.data_gen.query_document_encoder import QueryDocumentEncoderI
from tlm.data_gen.rank_common import QueryDocPairInstance, encode_query_doc_pair_instance, join_two_tokens, \
    join_two_input_ids


class PairwiseQueryDocGenFromTokensList(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 max_seq_length):
        self.resource = resource
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_parsing = []
        ticker = TimeEstimator(len(qids))
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                continue
            ticker.tick()
            tokens_d: Dict[str, Tuple[List[str], List[List[str]]]] = self.resource.get_doc_tokens_d(qid)
            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.resource, qid)
            q_tokens = self.resource.get_q_tokens(qid)

            def iter_passages(doc_id):
                title, body = tokens_d[doc_id]
                # Pair of [Query, Content]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, title, body)
                return insts

            for pos_doc_id in pos_doc_id_list:
                sampled_neg_doc_id = pick1(neg_doc_id_list)
                try:
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            q1, d1 = passage1
                            q2, d2 = passage2
                            for t1, t2 in zip(q1, q2):
                                assert t1 == t2
                            assert type(q1[0]) == str
                            assert type(d2[0]) == str
                            data_id = data_id_manager.assign({
                                'qid': qid,
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = QueryDocPairInstance(q1, d1, d2, data_id)
                            yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_parsing.append(qid)
                    if missing_cnt > 10:
                        print(missing_parsing)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[QueryDocPairInstance], out_path: str):
        def encode_fn(inst: QueryDocPairInstance):
            return encode_query_doc_pair_instance(self.tokenizer, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class QueryDocEntityGen(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 qid_to_entity_tokens: Dict[str, List[str]]
                 ):
        self.resource = resource
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.qid_to_entity_tokens: Dict[str, List[str]] = qid_to_entity_tokens

    def generate(self, data_id_manager, qids) -> Iterable[QueryEntityDocPairInstance]:
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        ticker = TimeEstimator(len(qids))
        for qid in qids:
            ticker.tick()
            if qid not in self.resource.get_doc_for_query_d():
                continue

            if qid not in self.qid_to_entity_tokens:
                continue

            tokens_d: Dict[str, Tuple[List[str], List[List[str]]]] = self.resource.get_doc_tokens_d(qid)
            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.resource, qid)
            q_tokens = self.resource.get_q_tokens(qid)
            entity_tokens = self.qid_to_entity_tokens[qid]
            assert len(entity_tokens) <= len(q_tokens)

            def iter_passages(doc_id):
                title, body = tokens_d[doc_id]
                # Pair of [Query, Content]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, title, body)
                # We will only use content parts
                return insts

            for pos_doc_id in pos_doc_id_list:
                sampled_neg_doc_id = pick1(neg_doc_id_list)
                try:
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            _, d1 = passage1
                            _, d2 = passage2
                            assert type(d2[0]) == str
                            data_id = data_id_manager.assign({
                                'qid': qid,
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = QueryEntityDocPairInstance(q_tokens, entity_tokens, d1, d2, data_id)
                            yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[QueryEntityDocPairInstance], out_path: str):
        def encode_fn(inst: QueryEntityDocPairInstance):
            return encode_query_entity_doc_pair_instance(self.tokenizer, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class QueryDocEntityConcatGen(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 max_seq_length,
                 q_max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]]
                 ):
        self.qde_gen = QueryDocEntityGen(resource, encoder, qid_to_entity_tokens)
        self.max_seq_length = max_seq_length
        self.q_max_seq_length = q_max_seq_length

    def generate(self, data_id_manager, qids) -> Iterable[QDEPaired]:
        iter: Iterable[QueryEntityDocPairInstance] = self.qde_gen.generate(data_id_manager, qids)
        for e in iter:
            def get_qde(doc_tokens):
                q_e = join_two_tokens(e.entity, e.query_tokens)
                d_e = join_two_tokens(e.entity, doc_tokens)
                return QDE(q_e, d_e)

            qde_paired = QDEPaired(
                get_qde(e.doc_tokens1),
                get_qde(e.doc_tokens2),
                e.data_id,
            )
            yield qde_paired

    def write(self, insts: List[QDEPaired], out_path: str):
        def encode_fn(inst: QDEPaired):
            return encode_concatenated_query_entity_doc_pair_instance(self.qde_gen.tokenizer,
                                                                      self.max_seq_length,
                                                                      self.q_max_seq_length,
                                                                      inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class QueryDocEntityConcatPointwisePredictionGen(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 max_seq_length,
                 q_max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]]
                 ):
        self.resource = resource
        self.encoder = encoder
        self.tokenizer = get_tokenizer()
        self.qid_to_entity_tokens: Dict[str, List[str]] = qid_to_entity_tokens
        self.max_seq_length = max_seq_length
        self.q_max_seq_length = q_max_seq_length

    def generate(self, data_id_manager, qids) -> Iterable[QDE_as_Ids]:
        self.success_docs = 0
        self.missing_cnt = 0
        self.missing_doc_qid = []
        for qid in TEL(qids):
            if qid not in self.resource.get_doc_for_query_d():
                continue

            if qid not in self.qid_to_entity_tokens:
                continue

            tokens_d: Dict[str, Tuple[List[str], List[List[str]]]] = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            q_tokens_ids = self.tokenizer.convert_tokens_to_ids(q_tokens)
            entity_tokens = self.qid_to_entity_tokens[qid]
            assert len(entity_tokens) <= len(q_tokens)

            def iter_passages(doc_id):
                title, body = tokens_d[doc_id]
                # Pair of [Query, Content]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, title, body)
                # We will only use content parts
                return insts
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    for passage_idx, passage in enumerate(iter_passages(doc_id)):
                        query_tokens, d_tokens = passage
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, ""),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                            'label': label,

                        })
                        entity_tokens: List[str] = self.qid_to_entity_tokens[qid]
                        d_tokens_ids = self.tokenizer.convert_tokens_to_ids(d_tokens)
                        entity_token_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
                        inst = QDE_as_Ids(join_two_input_ids(entity_token_ids, q_tokens_ids),
                                          join_two_input_ids(entity_token_ids, d_tokens_ids),
                                          label,
                                          data_id)
                        yield inst
                    self.success_docs += 1
                except KeyError:
                    self.missing_cnt += 1
                    self.missing_doc_qid.append(qid)
                    if self.missing_cnt > 10:
                        print(self.missing_doc_qid)
                        print("{} docs are missing while {} have suceed".format(self.missing_cnt, self.success_docs))
                        raise KeyError

    def write(self, insts: Iterable[QDE_as_Ids], out_path: str):
        def encode_fn(inst: QDE_as_Ids):
            return encode_qde_ids_instance(self.max_seq_length, self.q_max_seq_length, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class QDDistillGenI(ABC):
    @abstractmethod
    def generate(self, data_id_manager, qids, entries: List[Dict]):
        pass

    @abstractmethod
    def write(self, insts, out_path: str):
        pass


class QueryDocEntityDistilGen(QDDistillGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 max_seq_length,
                 q_max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]]
                 ):
        self.resource = resource
        self.max_seq_length = max_seq_length
        self.q_max_seq_length = q_max_seq_length
        self.qid_to_entity_tokens = qid_to_entity_tokens
        self.tokenizer = get_tokenizer()

    def generate(self, data_id_manager, qids, entries: List[Dict]) -> Iterable[QDE_as_Ids]:
        missing_cnt = 0
        success_docs = 0
        ticker = TimeEstimator(len(entries))

        def get_qrep_from_q_token_ids(q_ids):
            q_rep = " ".join(map(str, q_ids))
            return q_rep

        d_q_rep_to_qid = {}
        for qid in qids:
            q_tokens = self.resource.get_q_tokens(qid)
            q_ids = self.tokenizer.convert_tokens_to_ids(q_tokens)
            q_rep = get_qrep_from_q_token_ids(q_ids)
            d_q_rep_to_qid[q_rep] = qid

        qid_count = Counter()
        for e in entries:
            try:
                ticker.tick()
                q_tokens_ids: List[int] = e['q_tokens']
                q_rep = get_qrep_from_q_token_ids(q_tokens_ids)
                assert q_rep in d_q_rep_to_qid
                qid = d_q_rep_to_qid[q_rep]
                score = e['score']
                d_tokens_ids = e['d_tokens']
                if qid_count[qid] > 10 and score < 0:
                    continue
                qid_count[qid] += 1
                entity_tokens: List[str] = self.qid_to_entity_tokens[qid]
                entity_token_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
                data_id = data_id_manager.assign({
                    'qid': qid,
                    'score': str(score),
                })

                inst = QDE_as_Ids(join_two_input_ids(entity_token_ids, q_tokens_ids),
                                  join_two_input_ids(entity_token_ids, d_tokens_ids),
                                  score,
                                  data_id)
                yield inst
                success_docs += 1
            except KeyError:
                missing_cnt += 1
                if missing_cnt > 100 and missing_cnt % 100 == 1:
                    print("Fail rate {} of {}".format(missing_cnt, success_docs))

    def write(self, insts: Iterable[QDE_as_Ids], out_path: str):
        def encode_fn(inst: QDE_as_Ids):
            return encode_qde_ids_instance(self.max_seq_length, self.q_max_seq_length, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class FixedQTypeIDDistillGen(QDDistillGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyTokensListI,
                 max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]],
                 qid_to_qtype_id: Dict[str, int],
                 ):
        self.resource = resource
        self.max_seq_length = max_seq_length
        self.qid_to_entity_tokens = qid_to_entity_tokens
        self.qid_to_qtype_id = qid_to_qtype_id
        self.tokenizer = get_tokenizer()

    def generate(self, data_id_manager, qids, entries: List[Dict]) -> Iterable[QTypeDE_as_Ids]:
        missing_cnt = 0
        success_docs = 0
        ticker = TimeEstimator(len(entries))

        def get_qrep_from_q_token_ids(q_ids):
            q_rep = " ".join(map(str, q_ids))
            return q_rep

        d_q_rep_to_qid = {}
        for qid in qids:
            q_tokens = self.resource.get_q_tokens(qid)
            q_ids = self.tokenizer.convert_tokens_to_ids(q_tokens)
            q_rep = get_qrep_from_q_token_ids(q_ids)
            d_q_rep_to_qid[q_rep] = qid

        qid_count = Counter()
        for e in entries:
            try:
                ticker.tick()
                q_tokens_ids: List[int] = e['q_tokens']
                q_rep = get_qrep_from_q_token_ids(q_tokens_ids)
                assert q_rep in d_q_rep_to_qid
                qid = d_q_rep_to_qid[q_rep]
                score = e['score']
                d_tokens_ids = e['d_tokens']
                if qid_count[qid] > 10 and score < 0:
                    continue
                qid_count[qid] += 1
                entity_tokens: List[str] = self.qid_to_entity_tokens[qid]
                qtype_id: int = self.qid_to_qtype_id[qid]
                entity_token_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
                data_id = data_id_manager.assign({
                    'qid': qid,
                    'qtype_id': qtype_id,
                    'score': str(score),
                })

                inst = QTypeDE_as_Ids(qtype_id,
                                  join_two_input_ids(entity_token_ids, d_tokens_ids),
                                  score,
                                  data_id)
                yield inst
                success_docs += 1
            except KeyError:
                missing_cnt += 1
                if missing_cnt > 100 and missing_cnt % 100 == 1:
                    print("Fail rate {} of {}".format(missing_cnt, success_docs))

    def write(self, insts: Iterable[QTypeDE_as_Ids], out_path: str):
        def encode_fn(inst: QTypeDE_as_Ids):
            return encode_qtype_de_ids_instance(self.max_seq_length, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)


class FixedQTypeIDPredictionGen(MMDGenI):
    def __init__(self,
                 resource: ProcessedResourceTitleBodyTokensListI,
                 encoder: QueryDocumentEncoderI,
                 max_seq_length,
                 qid_to_entity_tokens: Dict[str, List[str]],
                 qid_to_qtype_id: Dict[str, int]
                 ):
        self.resource = resource
        self.tokenizer = get_tokenizer()
        self.qid_to_entity_tokens: Dict[str, List[str]] = qid_to_entity_tokens
        self.max_seq_length = max_seq_length
        self.qid_to_qtype_id = qid_to_qtype_id
        self.encoder = encoder

    def generate(self, data_id_manager, qids) -> Iterable[QTypeDE_as_Ids]:
        success_docs = 0
        missing_cnt = 0
        missing_doc_qid = []
        for qid in TEL(qids):
            if qid not in self.resource.get_doc_for_query_d():
                continue

            if qid not in self.qid_to_entity_tokens:
                continue

            tokens_d: Dict[str, Tuple[List[str], List[List[str]]]] = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            entity_tokens = self.qid_to_entity_tokens[qid]
            assert len(entity_tokens) <= len(q_tokens)

            def iter_passages(doc_id):
                title, body = tokens_d[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, title, body)
                return insts

            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    for passage_idx, passage in enumerate(iter_passages(doc_id)):
                        _, d_tokens = passage
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, ""),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                            'label': label,
                        })
                        qtype_id: int = self.qid_to_qtype_id[qid]
                        entity_tokens: List[str] = self.qid_to_entity_tokens[qid]
                        d_tokens_ids = self.tokenizer.convert_tokens_to_ids(d_tokens)
                        entity_token_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
                        inst = QTypeDE_as_Ids(qtype_id,
                                          join_two_input_ids(entity_token_ids, d_tokens_ids),
                                          label,
                                          data_id)
                        yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: Iterable[QTypeDE_as_Ids], out_path: str):
        def encode_fn(inst: QTypeDE_as_Ids):
            return encode_qtype_de_ids_instance(self.max_seq_length, inst)
        return write_records_w_encode_fn(out_path, encode_fn, insts)
