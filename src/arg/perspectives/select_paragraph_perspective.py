import os
import pickle
import string
from typing import List, Set, Iterable, Callable

import math
import nltk

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.basic_analysis import load_data_point
from arg.perspectives.declaration import ParagraphClaimPersFeature, PerspectiveCandidate
from arg.perspectives.ranked_list_interface import StaticRankedListInterface
from arg.pf_common.base import Paragraph, ScoreParagraph
from arg.pf_common.select_paragraph import subword_tokenize_functor, enum_paragraph
from arg.pf_common.text_processing import re_tokenize
from data_generator.subword_translate import Subword
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc, BertTokenizedCluewebDoc
from galagos.types import GalagoDocRankEntry
from list_lib import lfilter, lmap, flatten


def select_paragraph_dp_list(ci: StaticRankedListInterface,
                             clue12_13_df,
                             paragraph_iterator: Callable[[GalagoDocRankEntry], Iterable[Paragraph]],
                             datapoint_list: List[PerspectiveCandidate]) -> List[ParagraphClaimPersFeature]:
    not_found_set = set()

    cdf = 50 * 1000 * 1000

    ONE_PARA_PER_DOC = 1
    option = ONE_PARA_PER_DOC

    def paragraph_scorer(paragraph: Paragraph, cp_tokens: Set[str]) -> ScoreParagraph:
        paragraph_terms = set(paragraph.tokens)
        mentioned_terms = lfilter(lambda x: x in paragraph_terms, cp_tokens)
        mentioned_terms = re_tokenize(mentioned_terms)

        def idf(term: str):
            if term not in clue12_13_df:
                if term in string.printable:
                    return 0
                not_found_set.add(term)

            return math.log((cdf+0.5)/(clue12_13_df[term]+0.5))

        score = sum(lmap(idf, mentioned_terms))
        max_score = sum(lmap(idf, cp_tokens))
        return ScoreParagraph(paragraph=paragraph, score=score)

    def select_paragraph_from_datapoint(x: PerspectiveCandidate) -> ParagraphClaimPersFeature:
        ranked_docs: List[GalagoDocRankEntry] = ci.fetch(x.cid, x.pid)
        ranked_docs = ranked_docs[:100]
        cp_tokens = nltk.word_tokenize(x.claim_text) + nltk.word_tokenize(x.p_text)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        cp_tokens = set(cp_tokens)

        #  prefetch tokens and bert tokens
        doc_ids = lmap(lambda x: x.doc_id, ranked_docs)
        preload_man.preload(TokenizedCluewebDoc, doc_ids)
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)

        def paragraph_scorer_local(p):
            return paragraph_scorer(p, cp_tokens)

        def get_best_paragraph_from_doc(doc: GalagoDocRankEntry) -> List[ScoreParagraph]:
            paragraph_list = paragraph_iterator(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            score_paragraph.sort(key=lambda p: p.score, reverse=True)
            return score_paragraph[:1]

        def get_all_paragraph_from_doc(doc: GalagoDocRankEntry) -> List[ScoreParagraph]:
            paragraph_list = paragraph_iterator(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            return score_paragraph

        if option:
            get_paragraphs = get_best_paragraph_from_doc
        else:
            get_paragraphs = get_all_paragraph_from_doc

        candidate_paragraph: Iterable[ScoreParagraph] = flatten(lmap(get_paragraphs, ranked_docs))
        return ParagraphClaimPersFeature(claim_pers=x, feature=list(candidate_paragraph))

    def select_paragraph_from_datapoint_wrap(x: PerspectiveCandidate):
        try:
            return select_paragraph_from_datapoint(x)
        except KeyError as e:
            print(e)
        return None

    r = lmap(select_paragraph_from_datapoint_wrap, datapoint_list)

    r = list([e for e in r if e is not None])
    return r


class SelectParagraphWorker:
    def __init__(self, split, q_config_id, out_dir):
        self.out_dir = out_dir
        self.ci = StaticRankedListInterface(q_config_id)
        print("load__data_point")
        self.all_data_points = load_data_point(split)
        print("Load term stat")
        _, clue12_13_df = load_clueweb12_B13_termstat()
        self.clue12_13_df = clue12_13_df
        self.tokenizer = get_tokenizer()

    def paragraph_iterator(self, doc):
        def subword_tokenize(w: str) -> List[Subword]:
            return subword_tokenize_functor(self.tokenizer, w)
        step_size = 100
        subword_len = 350
        return enum_paragraph(step_size, subword_len, subword_tokenize, doc)

    def work(self, job_id):
        step = 10
        st = job_id * step
        ed = (job_id + 1) * step

        print("select paragraph")
        todo = self.all_data_points[st:ed]
        features: List[ParagraphClaimPersFeature] = select_paragraph_dp_list(
            self.ci,
            self.clue12_13_df,
            self.paragraph_iterator,
            todo)

        n_suc = len(features)
        n_all = len(todo)
        if n_all > n_suc:
            print("{} of {} succeed".format(n_suc, n_all))

        pickle.dump(features, open(os.path.join(self.out_dir, str(job_id)), "wb"))
