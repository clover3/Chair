import json
import os
from typing import List, Tuple, Any, Iterable

from nltk import sent_tokenize

from arg.qck.decl import QCKQuery, QCKCandidate
from cache import load_pickle_from
from cpath import at_data_dir
from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.analyze_code.doc_passage_join import lcs_based_join
from dataset_specific.msmarco.common import MSMarcoDoc, load_per_query_docs, load_query_group
from epath import job_man_dir
from list_lib import left
from misc_lib import exist_or_mkdir, DataIDManager, tprint
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, write_with_classification_instance_with_id
from tlm.data_gen.doc_encode_common import join_tokens, split_by_window
from tlm.data_gen.msmarco_doc_gen.misc_common import get_pos_neg_doc_ids_for_qid
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


class JoinGiveUp(Exception):
    pass


class TokensListPoolerWLabel:
    def __init__(self, tokens_list: List[Tuple[List[Any], int]]):
        self.tokens_list = tokens_list
        self.cursor = 0
        self.remaining_tokens = []
        self.last_label = 0

    def pool(self, max_acceptable_size, slice_if_too_long) -> Tuple[List[str], int]:
        if not self.remaining_tokens:
            cur_tokens, label = self.tokens_list[self.cursor]
        else:
            cur_tokens = self.remaining_tokens
            label = self.last_label

        if len(cur_tokens) <= max_acceptable_size:
            if self.remaining_tokens:
                self.remaining_tokens = []
            self.cursor += 1
            return cur_tokens, label
        elif slice_if_too_long:
            if label:
                raise JoinGiveUp()
            head = cur_tokens[:max_acceptable_size]
            tail = cur_tokens[max_acceptable_size:]
            self.remaining_tokens = tail
            self.last_label = label
            return head
        else:
            return [], label

    def is_end(self):
        return self.cursor >= len(self.tokens_list)


def enum_passage(title_tokens: List[Any], body_tokens: List[Tuple[List[Any], int]],
                       window_size: int, too_short_len: int) -> Iterable[Tuple[List[Any], int]]:
    pooler = TokensListPoolerWLabel(body_tokens)

    while not pooler.is_end():
        n_sent = 0
        tokens = []
        tokens.extend(title_tokens)
        current_window_size = window_size
        available_length = current_window_size - len(tokens)
        assert current_window_size > 10
        new_sent_tokens, label = pooler.pool(available_length, n_sent == 0)
        cur_label = label
        tokens.extend(new_sent_tokens)
        while not pooler.is_end() and len(tokens) < too_short_len:
            available_length = current_window_size - len(tokens)
            new_sent_tokens, label = pooler.pool(available_length, False)
            if label:
                cur_label = label
            if not new_sent_tokens:
                break
            tokens.extend(new_sent_tokens)
        yield tokens, cur_label


class PassageContainAssureEncoder:
    def __init__(self, max_seq_length, seg_selection_fn=None, max_seg_per_doc=999999):
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer()
        self.title_token_max = 64
        self.query_token_max = 64
        self.enum_passage_fn = enum_passage

        self.seg_selection_fn = seg_selection_fn
        self.max_seg_per_doc = max_seg_per_doc

    def encode(self, query_tokens, title, body, passage_text, p_loc) -> List[Tuple[List, List]]:
        query_tokens = query_tokens[:self.query_token_max]
        maybe_passage_end_loc = p_loc + len(passage_text) + 10

        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 0

        title_tokens = self.tokenizer.tokenize(title)
        title_tokens = title_tokens[:self.title_token_max]

        maybe_max_body_len = content_len - len(title_tokens)
        body_tokens: List[Tuple[List[str], int]] = self.get_tokens_sent_grouped(body, maybe_max_body_len, p_loc, maybe_passage_end_loc)
        ##
        if not body_tokens:
            raise JoinGiveUp()

        n_tokens = sum(map(len, left(body_tokens)))
        insts = []
        too_short_len = 40
        for idx, (second_tokens, label) in enumerate(self.enum_passage_fn(title_tokens, body_tokens, content_len, too_short_len)):
            out_tokens, segment_ids = join_tokens(query_tokens, second_tokens)
            entry = out_tokens, segment_ids
            insts.append((entry, label))
            assert idx <= n_tokens
            if idx >= self.max_seg_per_doc:
                break

        if self.seg_selection_fn is not None:
            insts = self.seg_selection_fn(insts)

        return insts

    def get_tokens_sent_grouped(self, body, maybe_max_body_len, p_start, p_end) -> List[Tuple[List[str], int]]:
        body_sents = sent_tokenize(body)
        out_body_tokens: List[Tuple[List[str], int]] = []
        assert p_start <= p_end
        cur_tokens = []
        last_end = 0
        is_middle_of_passage = False
        for sent in body_sents:
            tokens = self.tokenizer.tokenize(sent)

            st = body.find(sent, last_end)
            ed = st + len(sent)
            last_end = ed
            if not st <= ed:
                raise JoinGiveUp()
            if st < 0 :
                raise JoinGiveUp()

            if is_middle_of_passage:
                cur_tokens.extend(tokens)
                if p_end <= ed:
                    is_middle_of_passage = False
                    out_body_tokens.append((cur_tokens, 1))
                    cur_tokens = []
            else:
                # if current sent contains passage
                if st <= p_start and p_end <= ed:
                    if len(tokens) >= maybe_max_body_len:
                        raise JoinGiveUp()
                    else:
                        out_body_tokens.append((tokens, 1))
                    assert not cur_tokens
                # if current sent contains start, but not end
                elif st <= p_start <= ed:
                    assert not cur_tokens
                    is_middle_of_passage = True
                    cur_tokens.extend(tokens)
                elif st <= p_start and ed <= p_start:
                    out_body_tokens.append((tokens, 0))
                elif p_end <= st and p_end <= ed:
                    out_body_tokens.append((tokens, 0))
                else:
                    print(st, ed)
                    print(p_start, p_end)
                    assert False

        out_body_tokens_new: List[Tuple[List[str], int]] = []
        for tokens, label in out_body_tokens:
            if len(tokens) >= maybe_max_body_len:
                if not label:
                    for sub_tokens in split_by_window(tokens, maybe_max_body_len):
                        out_body_tokens_new.append((sub_tokens, label))
                else:
                    raise JoinGiveUp()
            else:
                out_body_tokens_new.append((tokens, label))

        return out_body_tokens_new


class SegmentPrediction:
    def __init__(self, resource: ProcessedResourceI,
                 basic_encoder,
                 max_seq_length):
        self.resource = resource
        self.encoder = basic_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, data_id_manager, qids, passage_dict_per_job, passage_qrels):
        for qid in qids:
            try:
                passages = []
                for passage_id, score in passage_qrels[qid].items():
                    if passage_id in passage_dict_per_job:
                        passages.append((passage_id, passage_dict_per_job[passage_id]))

                if not passages:
                    continue

                docs: List[MSMarcoDoc] = load_per_query_docs(qid, None)
                docs_d = {d.doc_id: d for d in docs}
                pos_doc_id_list, neg_doc_id_list \
                    = get_pos_neg_doc_ids_for_qid(self.resource, qid)
                q_tokens = self.resource.get_q_tokens(qid)

                def iter_passages(doc_id, passage_text, p_loc):
                    doc = docs_d[doc_id]
                    insts: List[Tuple[Tuple[List, List], int]]\
                        = self.encoder.encode(q_tokens, doc.title, doc.body, passage_text, p_loc)
                    return insts

                todo = []
                for pos_doc_id in pos_doc_id_list:
                    doc = docs_d[pos_doc_id]

                    for pid, p_text in passages:
                        p_loc = lcs_based_join(p_text, doc.body)
                        if p_loc >= 0:
                            todo.append((pos_doc_id, p_text, p_loc))
                for doc_id, p_text, p_loc in todo:
                    try:
                        for passage_idx, (passage, label) in enumerate(iter_passages(doc_id, p_text, p_loc)):

                            tokens_seg, seg_ids = passage
                            assert type(tokens_seg[0]) == str
                            assert type(seg_ids[0]) == int
                            data_id = data_id_manager.assign({
                                'query': QCKQuery(qid, ""),
                                'candidate': QCKCandidate(doc_id, ""),
                                'passage_idx': passage_idx,
                                'label': label
                            })
                            inst = ClassificationInstanceWDataID(tokens_seg, seg_ids, label, data_id)
                            yield inst
                    except JoinGiveUp:
                        print(JoinGiveUp, qid, doc_id)
            except KeyError as e:
                print(e)
                pass

    def write(self, insts: List[ClassificationInstanceWDataID], out_path: str):
        return write_with_classification_instance_with_id(self.tokenizer, self.max_seq_length, insts, out_path)


class MMDWorker(WorkerInterface):
    def __init__(self, split, generator, out_dir):
        self.out_dir = out_dir
        query_group = load_query_group(split)
        msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.{}.tsv".format(split))
        passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)
        self.split = split
        self.query_group = query_group
        self.generator = generator
        self.passage_qrels = passage_qrels
        self.info_dir = os.path.join(self.out_dir + "_info")
        exist_or_mkdir(self.info_dir)

    def work(self, job_id):
        qids = self.query_group[job_id]
        data_bin = 1000000
        data_id_st = job_id * data_bin
        data_id_ed = data_id_st + data_bin
        data_id_manager = DataIDManager(data_id_st, data_id_ed)
        tprint("generating instances")

        passage_dict_per_job = load_pickle_from(os.path.join(job_man_dir, "passage_join_for_{}".format(self.split), str(job_id)))
        insts = self.generator.generate(data_id_manager, qids, passage_dict_per_job, self.passage_qrels)
        # tprint("{} instances".format(len(insts)))
        out_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, out_path)

        info_path = os.path.join(self.info_dir, "{}.info".format(job_id))
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))