from typing import List, Iterator

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, MSMarcoDoc
from list_lib import lmap
from tlm.data_gen.adhoc_sent_tokenize import enum_passage_random_short
from tlm.data_gen.doc_encode_common import split_by_window
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SegmentRepresentation, SRPerQueryDoc, SRPerQuery
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceI
from nltk import sent_tokenize


class SegResourceMaker:
    def __init__(self,
                 resource: ProcessedResourceI,
                 max_seq_length,
                 max_seg_per_doc):
        self.resource = resource
        self.tokenizer = get_tokenizer()
        self.title_token_max = 64
        self.query_token_max = 64
        self.max_seq_length = max_seq_length
        self.max_seg_per_doc = max_seg_per_doc

    def enum_segment_rep(self,
                         query_tokens,
                         title_tokens: List[str],
                         body_tokens: List[List[str]],
                         content_len: int) -> Iterator[SegmentRepresentation]:
        query_tokens_in_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)

        for tokens in enum_passage_random_short(title_tokens, body_tokens, content_len):
            second_seg_token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            sl = SegmentRepresentation(query_tokens_in_ids, second_seg_token_ids)
            yield sl

    def generate(self, qids) -> Iterator[SRPerQuery]:
        for qid in qids:
            query_tokens = self.resource.get_q_tokens(qid)
            content_len = self.max_seq_length - 3 - len(query_tokens)
            try:
                docs: List[MSMarcoDoc] = load_per_query_docs(qid, None)
                docs_d = {d.doc_id: d for d in docs}

                sr_per_query_doc_list = []
                for doc_id in self.resource.get_doc_for_query_d()[qid]:
                    label = self.resource.get_label(qid, doc_id)
                    try:
                        doc = docs_d[doc_id]
                        segs: List[SegmentRepresentation] = self.get_segs(query_tokens, doc, content_len)
                        sr_per_query_doc = SRPerQueryDoc(doc_id, segs, label)
                        sr_per_query_doc_list.append(sr_per_query_doc)
                    except KeyError:
                        pass

                sr_per_query = SRPerQuery(qid, sr_per_query_doc_list)
                yield sr_per_query
            except KeyError as e:
                print(e)
                print(doc_id)
            except FileNotFoundError as e:
                print(e)
                print(qid)

    def get_segs(self, query_tokens, doc, content_len) -> List[SegmentRepresentation]:
        title_tokens = self.tokenizer.tokenize(doc.title)
        title_tokens = title_tokens[:self.title_token_max]
        maybe_max_body_len = content_len - len(title_tokens)
        body_tokens_list: List[List[str]] = self.get_tokens_sent_grouped(doc.body, maybe_max_body_len)
        if not body_tokens_list:
            body_tokens_list: List[List[str]] = [["[PAD]"]]
        n_tokens = sum(map(len, body_tokens_list))
        segs = []
        for idx, seg in enumerate(self.enum_segment_rep(query_tokens, title_tokens, body_tokens_list, content_len)):
            segs.append(seg)
            assert idx <= n_tokens
            if idx >= self.max_seg_per_doc:
                break

        return segs

    def get_tokens_sent_grouped(self, body, maybe_max_body_len) -> List[List[str]]:
        body_sents = sent_tokenize(body)
        body_tokens: List[List[str]] = lmap(self.tokenizer.tokenize, body_sents)
        body_tokens: List[List[str]] = list([tokens for tokens in body_tokens if tokens])
        out_body_tokens: List[List[str]] = []
        for tokens in body_tokens:
            if len(tokens) >= maybe_max_body_len:
                out_body_tokens.extend(split_by_window(tokens, maybe_max_body_len))
            else:
                out_body_tokens.append(tokens)

        return out_body_tokens

