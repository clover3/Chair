from collections import Counter

from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.qck.decl import QCKQuery, QCKCandidate
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap, get_max_idx, flatten, lflatten
from misc_lib import find_max_idx, TEL
from tlm.data_gen.classification_common import ClassificationInstanceWDataID, write_with_classification_instance_with_id
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti, ProcessedResourceMultiInterface
from typing import List, Iterable, Callable, Dict, Tuple, Set


def regroup_sent_list(tokens_list: List[List[str]], n) -> List[List[str]]:
    output = []
    i = 0
    while i < len(tokens_list):
        output.append(lflatten(tokens_list[i:i+n]))
        i += n
    return output


class MaxSentEncoder:
    def __init__(self, bm25_module, max_seq_length):
        self.max_seq_length = max_seq_length
        self.bm25_module = bm25_module
        pc_tokenize = PCTokenizer()
        self.tokenize_stem = pc_tokenize.tokenize_stem
        bert_tokenizer = get_tokenizer()
        self.bert_tokenize = bert_tokenizer.tokenize

    def encode(self, query_text,
                     stemmed_title_tokens: List[str],
                     stemmed_body_tokens_list: List[List[str]],
                     bert_title_tokens: List[str],
                     bert_body_tokens_list: List[List[str]]) -> List[Tuple[List, List]]:

        # Title and body sentences are trimmed to 64 * 5 chars
        # score each sentence based on bm25_module
        stemmed_query_tokens = self.bert_tokenize(query_text)
        q_tf = Counter(stemmed_query_tokens)
        assert len(stemmed_body_tokens_list) == len(bert_body_tokens_list)

        stemmed_body_tokens_list = regroup_sent_list(stemmed_body_tokens_list, 4)
        bert_body_tokens_list = regroup_sent_list(bert_body_tokens_list, 4)

        def get_score(sent_idx):
            doc_tf = Counter(stemmed_body_tokens_list[sent_idx])
            return self.bm25_module.score_inner(q_tf, doc_tf)

        bert_query_tokens = self.bert_tokenize(query_text)
        if stemmed_body_tokens_list:
            seg_scores = lmap(get_score, range(len(stemmed_body_tokens_list)))
            max_idx = get_max_idx(seg_scores)
            content_len = self.max_seq_length - 3 - len(bert_query_tokens)
            second_tokens = bert_body_tokens_list[max_idx][:content_len]
        else:
            second_tokens = []
        out_tokens = ["[CLS]"] + bert_query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
        segment_ids = [0] * (len(bert_query_tokens) + 2) + [1] * (len(second_tokens) + 1)
        entry = out_tokens, segment_ids
        return [entry]


class BestSentGen:
    def  __init__(self, resource: ProcessedResourceMultiInterface,
                 basic_encoder,
                 max_seq_length,
                 qck_stlye_info=True):
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

            query_text = self.resource.get_query_text(qid)
            bert_tokens_d = self.resource.get_bert_tokens_d(qid)
            stemmed_tokens_d = self.resource.get_stemmed_tokens_d(qid)
            for doc_id in self.resource.get_doc_for_query_d()[qid]:
                label = self.resource.get_label(qid, doc_id)
                try:
                    bert_title_tokens, bert_body_tokens_list = bert_tokens_d[doc_id]
                    stemmed_title_tokens, stemmed_body_tokens_list = stemmed_tokens_d[doc_id]
                    insts: List[Tuple[List, List]]\
                        = self.encoder.encode(
                            query_text,
                            stemmed_title_tokens,
                            stemmed_body_tokens_list,
                            bert_title_tokens,
                            bert_body_tokens_list
                    )

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

