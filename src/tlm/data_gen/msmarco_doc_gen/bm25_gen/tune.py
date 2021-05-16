import os
import pickle
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple, Any

from data_generator.job_runner import WorkerInterface, JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_token_d_sent_level, load_token_d_bm25_tokenize
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.adhoc_sent_tokenize import split_by_window
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, \
    write_with_classification_instance_with_id, PairedInstance
from tlm.data_gen.doc_encode_common import draw_window_size
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI, ProcessedResource, \
    ProcessedResource10doc


class PassageSegment(NamedTuple):
    bert_tokens: List[str]
    bm25_tokens: List[Any]


class DocEncoder:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer()
        self.title_token_max = 64
        self.query_token_max = 64

    def encode(self,
                 q_tokens,
                 bert_title_tokens,
                 bert_doc_tokens,
                 bm25_title_tokens,
                 bm25_doc_tokens,
                 no_short=False
                 ) -> List[PassageSegment]:
        doc_align_map = None
        if len(bm25_doc_tokens) != len(bert_doc_tokens):
            assert False

        def clip_length(bert_tokens, space_tokens, n_max_tokens):
            if len(bert_tokens) > n_max_tokens:
                return count_space_tokens(bert_tokens)
            else:
                return len(space_tokens)

        clipped_title_len = clip_length(bert_title_tokens, bm25_title_tokens, self.title_token_max)
        bm25_title_tokens = bm25_title_tokens[:clipped_title_len]
        bert_title_tokens = bert_title_tokens[:self.title_token_max]

        q_tokens = q_tokens[:self.query_token_max]
        content_len = self.max_seq_length - 3 - len(q_tokens)
        window_size = content_len
        body_tokens = bert_doc_tokens
        cursor = 0
        def convert_cursor(cursor):
            if doc_align_map is None:
                return cursor
            else:
                return doc_align_map[cursor]

        passage_segment_list: List[PassageSegment] = []
        resplit_sents = None
        in_virtual_split = False
        virtual_cursor = 0
        def get_current_token():
            if not in_virtual_split:
                return body_tokens[cursor]
            else:
                return resplit_sents[virtual_cursor]

        def move_token_cursor():
            nonlocal cursor, virtual_cursor, in_virtual_split
            if not in_virtual_split:
                cursor += 1
            else:
                virtual_cursor += 1
                if virtual_cursor == len(resplit_sents):
                    in_virtual_split = False

        while cursor < len(body_tokens):
            bert_tokens: List[str] = []
            bert_tokens.extend(bert_title_tokens)
            bm25_tokens: List[List] = []
            bm25_tokens.append(bm25_title_tokens)
            any_sent_added = False
            if no_short:
                current_window_size = window_size
            else:
                current_window_size = draw_window_size(window_size)
            maybe_max_body_len = current_window_size - len(bert_title_tokens)
            #
            # if len(body_tokens[cursor]) >= maybe_max_body_len:
            #     tokens_list = split_by_window(body_tokens[cursor], 40)
            #     virtual_cursor = 0
            #     resplit_sents = tokens_list
            #     in_virtual_split = True

            assert current_window_size > 10
            while cursor < len(body_tokens) and \
                    len(bert_tokens) + len(get_current_token()) <= current_window_size:

                bert_tokens.extend(get_current_token())
                cursor_for_bm25_lines = convert_cursor(cursor)
                bm25_tokens.append(bm25_doc_tokens[cursor_for_bm25_lines])
                move_token_cursor()
                any_sent_added = True

            if not any_sent_added and cursor < len(body_tokens) and \
                    len(bert_tokens) + len(get_current_token()) > current_window_size:
                raise LongSentenceError()


            ps = PassageSegment(bert_tokens, bm25_tokens)
            passage_segment_list.append(ps)
        return passage_segment_list


class DocRep(NamedTuple):
    qid: str
    doc_id: str
    passage_segment_list: List[PassageSegment]
    label: str

class LongSentenceError(Exception):
    pass


class PointwiseGen:
    def  __init__(self, prcessed_resource: ProcessedResourceI,
                        get_tokens_d_bert,
                        get_tokens_d_bm25,
                        parallel_encoder,
                        max_seq_length):
        self.prcessed_resource = prcessed_resource
        self.get_tokens_d_bert = get_tokens_d_bert
        self.get_tokens_d_bm25 = get_tokens_d_bm25
        self.encoder = parallel_encoder
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length

    def generate(self, qids) -> List[Tuple[str, List[DocRep]]]:
        missing_cnt = 0
        success_docs = 0
        missing_doc_qid = []
        segmented_entries: List[Tuple[str, List[DocRep]]] = []
        for qid in qids:
            print(qid)
            if qid not in self.prcessed_resource.get_doc_for_query_d():
                continue
            bert_tokens_d = self.get_tokens_d_bert(qid)
            q_tokens = self.prcessed_resource.get_q_tokens(qid)
            bm25_tokens_d = self.get_tokens_d_bm25(qid)
            docs = []
            doc_ids_todo = list(self.prcessed_resource.get_doc_for_query_d()[qid])
            fail_list = []
            for doc_id in doc_ids_todo:
                label = self.prcessed_resource.get_label(qid, doc_id)
                try:
                    bert_title_tokens, bert_doc_tokens = bert_tokens_d[doc_id]
                    bm25_title_tokens, bm25_doc_tokens = bm25_tokens_d[doc_id]
                    if doc_id in fail_list:
                        no_short = True
                    else:
                        no_short = False
                    passage_segment_list: List[PassageSegment] = self.encoder.encode(q_tokens,
                                                                       bert_title_tokens,
                                                                       bert_doc_tokens,
                                                                       bm25_title_tokens,
                                                                       bm25_doc_tokens,
                                                                       no_short
                                                                       )

                    doc_rep = DocRep(qid, doc_id, passage_segment_list, label)
                    docs.append(doc_rep)

                    success_docs += 1
                except LongSentenceError:
                    if doc_id not in fail_list:
                        doc_ids_todo.append(doc_id)
                        fail_list.append(doc_id)
                    else:
                        print("LongSentenceError")

                except KeyError:
                    missing_cnt += 1
                    missing_doc_qid.append(qid)
                    if missing_cnt > 10:
                        print(missing_doc_qid)
                        print("success: ", success_docs)
                        raise KeyError

            per_query = qid, docs
            segmented_entries.append(per_query)
        return segmented_entries


def count_space_tokens(tokens):
    cnt = 0
    for t in tokens:
        if not t[0] == "#":
            cnt += 1
    return cnt


class MMDWorker(WorkerInterface):
    def __init__(self, query_group, generator, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.generator = generator

    def work(self, job_id):
        qids = self.query_group[job_id]
        segmented_entries = self.generator.generate(qids)
        out_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(segmented_entries,
                    open(out_path, "wb")
                    )


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    max_seq_length = 512

    def get_tokens_d_bert(qid):
        return load_token_d_sent_level(split, qid)

    def get_tokens_d_bm25(qid):
        return load_token_d_bm25_tokenize(split, qid)

    basic_encoder = DocEncoder(max_seq_length)
    generator = PointwiseGen(resource,
                             get_tokens_d_bert,
                             get_tokens_d_bm25,
                             basic_encoder,
                             max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir,
                       len(resource.query_group)-1,
                       "MMD_bm25_segmentation_{}".format(split),
                       factory)
    runner.start()

