from typing import NamedTuple, List

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopicv2
from bert_api.doc_score_defs import DocumentScorerOutputSbword, DocumentScorerOutput
from bert_api.doc_score_helper import RemoteDocumentScorer
from data_generator.tokenize_helper import TokenizedText
from list_lib import lmap
from trainer.promise import MyFuture


class AnalyzedDocument(NamedTuple):
    doc_text: str
    claim_id: str
    p_ids: List[str]

    c_scores: DocumentScorerOutput
    # (window_id, idx of p) -> score
    p_scores: List[DocumentScorerOutput]
    # (window_id) -> score
    dot_tokens: List[str]  # space tokenized


def analyze_doc_wrt_ca_topic(document_scorer: RemoteDocumentScorer, topic: CaTopicv2, doc: TokenizedText) -> AnalyzedDocument:
    # For each segment, score toward the claim and perspectives.
    # Promise[DocumentScorerOutput]
    sdp: MyFuture = document_scorer.score_relevance(topic.claim_text, doc)
    p_list = [topic.target_p] + topic.other_ps

    # Scored document promise list
    sdp_list = []
    pids = []
    for p_group in p_list:
        pid0, p_text0 = p_group[0]
        scp = document_scorer.score_relevance(p_text0, doc)
        sdp_list.append(scp)
        pids.append(pid0)

    document_scorer.pk.do_duty()

    dso_claim: DocumentScorerOutputSbword = sdp.get()
    dso_claim2: DocumentScorerOutput = DocumentScorerOutput.from_dsos(dso_claim, doc)

    def sdp_to_dso(sdp: MyFuture) -> DocumentScorerOutput:
        dso_p = DocumentScorerOutput.from_dsos(sdp.get(), doc)
        return dso_p

    dso_p_list: List[DocumentScorerOutput] = lmap(sdp_to_dso, sdp_list)
    return AnalyzedDocument(doc.text, topic.cid, pids, dso_claim2, dso_p_list, doc.tokens)
