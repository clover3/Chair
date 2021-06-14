import os
import pickle
from typing import List, Tuple

from data_generator.job_runner import WorkerInterface, JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_token_d_sent_level, load_token_d_bm25_tokenize
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.bm25_gen.bm25_gen_common import PassageSegmentEnumerator, PassageSegment, DocRep, \
    LongSentenceError
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI, ProcessedResource10doc


class PointwiseGen:
    def __init__(self, prcessed_resource: ProcessedResourceI,
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

    basic_encoder = PassageSegmentEnumerator(max_seq_length)
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

