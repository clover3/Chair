import os
import pickle
import string
from collections import Counter
from typing import List, Iterable, Callable, NamedTuple

import math
import nltk
from arg.pf_common.select_paragraph import subword_tokenize_functor, enum_paragraph

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.pf_common.base import TPDataPoint, ParagraphFeature, Paragraph, ScoreParagraph, DPID
from arg.pf_common.ranked_list_interface import RankedListInterface
from arg.pf_common.text_processing import re_tokenize
from arg.ukp.data_loader import UkpDataPoint, load_all_data_flat
from cache import load_from_pickle
from data_generator.argmining.ukp_header import label_names
from data_generator.subword_translate import Subword
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc, BertTokenizedCluewebDoc
from galagos.types import SimpleRankedListEntry
from list_lib import lfilter, lmap, flatten
from misc_lib import ceil_divide


def ukp_datapoint_to_tp_datapoint(x: UkpDataPoint) -> TPDataPoint:
    label = label_names.index(x.label)
    return TPDataPoint(id=DPID(str(x.id)), text1=x.topic, text2=x.sentence, label=label)


def remove_duplicate(candidate_paragraph: List[ScoreParagraph]) -> List[ScoreParagraph]:
    r = []

    def list_equal(a: List[str], b: List[str]) -> bool:
        if len(a) != len(b):
            return False
        for a_e, b_e in zip(a,b):
            if a_e != b_e :
                return False
        return True

    for i, para in enumerate(candidate_paragraph):
        if r:
            prev_para: ScoreParagraph = r[-1]
            if list_equal(para.paragraph.tokens, prev_para.paragraph.tokens):
                pass
            else:
                r.append(para)
        else:
            r.append(para)
    return r

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


ONE_PARA_PER_DOC = 1
ANY_PARA_PER_DOC = 2

class Option(NamedTuple):
    para_per_doc: int

def select_paragraph_dp_list(ci: RankedListInterface,
                             dp_id_to_q_res_id_fn: Callable[[str], str],
                             paragraph_scorer_factory: Callable[[TPDataPoint], Callable[[Paragraph], ScoreParagraph]],
                             paragraph_iterator: Callable[[SimpleRankedListEntry], Iterable[Paragraph]],
                             datapoint_list: List[TPDataPoint],
                             option: Option,
                             ) -> List[ParagraphFeature]:

    n_passages = 20

    def select_paragraph_from_datapoint(x: TPDataPoint) -> ParagraphFeature:
        try:
            ranked_docs: List[SimpleRankedListEntry] = ci.fetch_from_q_res_id(dp_id_to_q_res_id_fn(x.id))
            ranked_docs = ranked_docs[:100]
        except KeyError:
            ranked_docs = []

        paragraph_scorer_local: Callable[[Paragraph], ScoreParagraph] = paragraph_scorer_factory(x)
        #  prefetch tokens and bert tokens
        doc_ids = lmap(lambda x: x.doc_id, ranked_docs)
        preload_man.preload(TokenizedCluewebDoc, doc_ids)
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)

        def get_best_paragraph_from_doc(doc: SimpleRankedListEntry) -> List[ScoreParagraph]:
            paragraph_list = paragraph_iterator(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            score_paragraph.sort(key=lambda p: p.score, reverse=True)
            return score_paragraph[:1]

        def get_all_paragraph_from_doc(doc: SimpleRankedListEntry) -> List[ScoreParagraph]:
            paragraph_list = paragraph_iterator(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            return score_paragraph

        if option.para_per_doc == ONE_PARA_PER_DOC:
            get_paragraphs = get_best_paragraph_from_doc
        else:
            get_paragraphs = get_all_paragraph_from_doc

        candidate_paragraph: List[ScoreParagraph] = list(flatten(lmap(get_paragraphs, ranked_docs)))
        candidate_paragraph.sort(key=lambda x: x.score, reverse=True)
        candidate_paragraph = remove_duplicate(candidate_paragraph)

        return ParagraphFeature(datapoint=x,
                                feature=candidate_paragraph[:n_passages])

    r = lmap(select_paragraph_from_datapoint, datapoint_list)
    return r


def build_dp_id_to_q_res_id_fn():
    d = load_from_pickle("ukp_10_dp_id_to_q_res_id")

    def fn(dp_id: str):
        return d[dp_id]

    return fn


class SelectParagraphWorker:
    def __init__(self, option, out_dir):
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
        self.option = option

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
            todo,
            self.option)

        pickle.dump(features, open(os.path.join(self.out_dir, str(job_id)), "wb"))
