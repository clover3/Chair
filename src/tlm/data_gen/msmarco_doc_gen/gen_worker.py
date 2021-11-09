from collections import Counter
from typing import List, Iterable, Dict, Tuple
from typing import OrderedDict

from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, print_dict_tab
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.adhoc_datagen import TitleRepeatInterface
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, \
    write_with_classification_instance_with_id, PairedInstance
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI, ProcessedResourcePredict, \
    ProcessedResourceTitleBodyPredict, ProcessedResourceTitleBodyI
from tlm.data_gen.pairwise_common import combine_features


class PointwiseGen(MMDGenI):
    def __init__(self, resource: ProcessedResourceI,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                # assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'doc_id': doc_id,
                            'passage_idx': passage_idx,
                            'label': label,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)


class GenerateFromTitleBody:
    def __init__(self, resource: ProcessedResourceTitleBodyI,
                 doc_encoder: TitleRepeatInterface,
                 max_seq_length):
        self.resource = resource
        self.doc_encoder = doc_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d: Dict[str, Tuple[List[str], List[str]]] = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    title_tokens, body_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.doc_encoder.encode(q_tokens, title_tokens, body_tokens)

                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'doc_id': doc_id,
                            'passage_idx': passage_idx,
                            'label': label,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)


class FirstPassagePairGenerator:
    def __init__(self, resource: ProcessedResourceI,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)

            pos_doc_id_list, neg_doc_id_list \
                = get_pos_neg_doc_ids_for_qid(self.resource, qid)

            def iter_passages(doc_id):
                doc_tokens = tokens_d[doc_id]
                insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

                for passage_idx, passage in enumerate(insts):
                    yield passage
            try:
                for pos_doc_id in pos_doc_id_list:
                    sampled_neg_doc_id = pick1(neg_doc_id_list)
                    for passage_idx1, passage1 in enumerate(iter_passages(pos_doc_id)):
                        for passage_idx2, passage2 in enumerate(iter_passages(sampled_neg_doc_id)):
                            tokens_seg1, seg_ids1 = passage1
                            tokens_seg2, seg_ids2 = passage2

                            data_id = data_id_manager.assign({
                                'doc_id1': pos_doc_id,
                                'passage_idx1': passage_idx1,
                                'doc_id2': sampled_neg_doc_id,
                                'passage_idx2': passage_idx2,
                            })
                            inst = PairedInstance(tokens_seg1, seg_ids1, tokens_seg2, seg_ids2, data_id)
                            yield inst
                    success_docs += 1
            except KeyError:
                missing_cnt += 1
                missing_doc_qid.append(qid)
                if missing_cnt > 10:
                    print(missing_doc_qid)
                    raise

    def write(self, insts: List[PairedInstance], out_path: str):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)

        try:
            length = len(insts)
        except TypeError:
            length = 0

        return write_records_w_encode_fn(out_path, encode_fn, insts, length)


class PassageLengthInspector:
    def __init__(self, resource: ProcessedResourceI,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        counter = Counter()
        cut_list = [50, 100, 200, 300, 500, 1000, 999999999]
        for qid in qids:
            if qid not in self.resource.get_doc_for_query_d():
                assert not self.resource.query_in_qrel(qid)
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)

            pos_doc_id_list = []
            neg_doc_id_list = []
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                if label:
                    pos_doc_id_list.append(doc_id)
                else:
                    neg_doc_id_list.append(doc_id)

            try:
                for pos_doc_id in pos_doc_id_list:
                    sampled_neg_doc_id = pick1(neg_doc_id_list)
                    pos_doc_tokens = tokens_d[pos_doc_id]
                    for cut in cut_list:
                        if len(pos_doc_tokens) < cut:
                            counter["pos_under_{}".format(cut)] += 1
                        else:
                            counter["pos_over_{}".format(cut)] += 1
                        neg_doc_tokens = tokens_d[sampled_neg_doc_id]
                        if len(neg_doc_tokens) < cut:
                            counter["neg_under_{}".format(cut)] += 1
                        else:
                            counter["neg_over_{}".format(cut)] += 1

                    inst = PairedInstance([], [], [], [], 0)
                    yield inst
                    success_docs += 1
            except KeyError:
                missing_cnt += 1
                missing_doc_qid.append(qid)
                if missing_cnt > 10 * 40:
                    print(missing_doc_qid)
                    raise

        for cut in cut_list:
            n_pos_short = counter["pos_under_{}".format(cut)]
            n_short = n_pos_short + counter["neg_under_{}".format(cut)]
            if n_short > 0:
                p_pos_if_short = n_pos_short / n_short
                print("P(Pos|Len<{})={}".format(cut, p_pos_if_short))
            else:
                print("P(Pos|Len<{})={}".format(cut, "div 0"))


        print_dict_tab(counter)

    def write(self, insts: Iterable[PairedInstance], out_path: str):
        cnt = 0
        for i in insts:
            cnt += 1


class PredictionAllPassageGenerator:
    def __init__(self, resource: ProcessedResourcePredict,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        for qid in qids:
            if qid not in self.resource.candidate_doc_d:
                assert qid not in self.resource.qrel.qrel_d
                continue

            tokens_d = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)
            for doc_id in self.resource.get_candidate_doc_d(qid):
                label = self.resource.get_label(qid, doc_id)
                try:
                    doc_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.encoder.encode(q_tokens, doc_tokens)

                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, ""),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        yield inst
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)




def memory_profile_print():
    from pympler import muppy, summary
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    # Prints out a summary of the large objects
    summary.print_(sum1)
    # Get references to certain types of objects such as dataframe
    dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
    for d in dataframes:
        print(d.columns.values)
        print(len(d))


class PredictionGenFromTitleBody(MMDGenI):
    def __init__(self, resource: ProcessedResourceTitleBodyPredict,
                 basic_encoder: TitleRepeatInterface,
                 max_seq_length):
        self.resource = resource
        self.doc_encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids):
        missing_cnt = 0
        success_docs = 0
        n_passage = 0
        for qid in qids:
            if qid not in self.resource.candidate_doc_d:
                assert qid not in self.resource.qrel.qrel_d
                continue

            tokens_d: Dict[str, Tuple[List, List]] = self.resource.get_doc_tokens_d(qid)
            q_tokens = self.resource.get_q_tokens(qid)

            data_size_maybe = 0
            for title_tokens, body_tokens in tokens_d.values():
                data_size_maybe += len(title_tokens)
                data_size_maybe += len(body_tokens)
            for doc_id in self.resource.candidate_doc_d[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    title_tokens, body_tokens = tokens_d[doc_id]
                    insts: List[Tuple[List, List]] = self.doc_encoder.encode(q_tokens, title_tokens, body_tokens)

                    for passage_idx, passage in enumerate(insts):
                        tokens_seg, seg_ids = passage
                        assert type(tokens_seg[0]) == str
                        assert type(seg_ids[0]) == int
                        data_id = data_id_manager.assign({
                            'query': QCKQuery(qid, ""),
                            'candidate': QCKCandidate(doc_id, ""),
                            'passage_idx': passage_idx,
                        })
                        inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                        n_passage += 1
                        yield inst
                        # if n_passage % 1000 == 0:
                        #     tprint("n_passage : {}".format(n_passage))
                        #     tprint('gc.get_count()', gc.get_count())
                        #     tprint('gc.get_stats', gc.get_stats())
                    success_docs += 1
                except KeyError:
                    missing_cnt += 1
                    if missing_cnt > 10:
                        print("success: ", success_docs)
                        raise KeyError
        print(" {} of {} has long title".format(self.doc_encoder.long_title_cnt, self.doc_encoder.total_doc_cnt))

    def write(self, insts: Iterable[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)


