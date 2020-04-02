import os
import pickle
import string
from collections import Counter
from typing import List, Iterable, Callable

import math
import nltk

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.pf_common.base import TPDataPoint, ParagraphFeature, Paragraph, ScoreParagraph
from arg.pf_common.ranked_list_interface import RankedListInterface
from arg.pf_common.select_paragraph import subword_tokenize_functor, enum_paragraph
from arg.pf_common.text_processing import re_tokenize
from arg.ukp.data_loader import UkpDataPoint, load_all_data_flat
from cache import load_from_pickle
from data_generator.argmining.ukp_header import label_names
from data_generator.common import get_tokenizer
from data_generator.subword_translate import Subword
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc, BertTokenizedCluewebDoc
from galagos.types import GalagoDocRankEntry
from list_lib import lfilter, lmap, flatten
from misc_lib import ceil_divide


def ukp_datapoint_to_tp_datapoint(x: UkpDataPoint) -> TPDataPoint:
    label = label_names.index(x.label)
    return TPDataPoint(id=str(x.id), text1=x.topic, text2=x.sentence, label=label)


def paragraph_scorer_factory_factory(clue12_13_df: Counter):
    def paragraph_scorer_factory(x: TPDataPoint) -> Callable[[Paragraph], ScoreParagraph]:
        not_found_set = set()
        cp_tokens = nltk.word_tokenize(x.text1) + nltk.word_tokenize(x.text2)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        cp_tokens = set(cp_tokens)
        cdf = 50 * 1000 * 1000

        def paragraph_scorer(paragraph: Paragraph) -> ScoreParagraph:
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

        return paragraph_scorer
    return paragraph_scorer_factory


def select_paragraph_dp_list(ci: RankedListInterface,
                             dp_id_to_q_res_id_fn: Callable[[str], str],
                             paragraph_scorer_factory: Callable[[TPDataPoint], Callable[[Paragraph], ScoreParagraph]],
                             paragraph_iterator: Callable[[GalagoDocRankEntry], Iterable[Paragraph]],
                             datapoint_list: List[TPDataPoint]) -> List[ParagraphFeature]:

    ONE_PARA_PER_DOC = 1
    option = ONE_PARA_PER_DOC

    def select_paragraph_from_datapoint(x: TPDataPoint) -> ParagraphFeature:
        ranked_docs: List[GalagoDocRankEntry] = ci.fetch_from_q_res_id(dp_id_to_q_res_id_fn(x.id))
        ranked_docs = ranked_docs[:100]

        paragraph_scorer_local: Callable[[Paragraph], ScoreParagraph] = paragraph_scorer_factory(x)
        #  prefetch tokens and bert tokens
        doc_ids = lmap(lambda x: x.doc_id, ranked_docs)
        preload_man.preload(TokenizedCluewebDoc, doc_ids)
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)

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
        return ParagraphFeature(datapoint=x, feature=list(candidate_paragraph))

    r = lmap(select_paragraph_from_datapoint, datapoint_list)
    return r


def build_dp_id_to_q_res_id_fn():
    d = load_from_pickle("ukp_10_dp_id_to_q_res_id")

    def fn(dp_id: str):
        return d[dp_id]

    return fn


class SelectParagraphWorker:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.ci = RankedListInterface()
        print("load__data_point")
        self.all_data_points: List[TPDataPoint] = lmap(ukp_datapoint_to_tp_datapoint, load_all_data_flat())
        self.data_step_size = 50

        total_jobs = ceil_divide(len(self.all_data_points), self.data_step_size)
        print("total_jobs :", total_jobs )
        print("Load term stat")
        _, clue12_13_df = load_clueweb12_B13_termstat()
        self.clue12_13_df = clue12_13_df
        self.dp_id_to_q_res_id_fn = build_dp_id_to_q_res_id_fn()
        self.tokenizer = get_tokenizer()

    def paragraph_iterator(self, doc):
        def subword_tokenize(w: str) -> List[Subword]:
            return subword_tokenize_functor(self.tokenizer, w)
        step_size = 100
        subword_len = 350
        return enum_paragraph(step_size, subword_len, subword_tokenize, doc)

    def work(self, job_id):
        step = self.data_step_size
        st = job_id * step
        ed = (job_id + 1) * step

        print("select paragraph")
        todo = self.all_data_points[st:ed]
        local_factory = paragraph_scorer_factory_factory(self.clue12_13_df)
        features: List[ParagraphFeature] = select_paragraph_dp_list(
            self.ci,
            build_dp_id_to_q_res_id_fn(),
            local_factory,
            self.paragraph_iterator,
            todo)

        pickle.dump(features, open(os.path.join(self.out_dir, str(job_id)), "wb"))
