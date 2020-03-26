import string
from typing import List, Iterable, Set, NamedTuple

import math
import nltk

from arg.perspectives.basic_analysis import PerspectiveCandidate
from arg.perspectives.build_feature import re_tokenize
from arg.perspectives.ranked_list_interface import StaticRankedListInterface
from data_generator.subword_translate import Subword
from datastore.interface import preload_man, load
from datastore.table_names import TokenizedCluewebDoc, BertTokenizedCluewebDoc
from galagos.parse import GalagoDocRankEntry
from list_lib import lmap, lfilter, flatten


class Paragraph(NamedTuple):
    doc_id: str
    doc_rank: int
    doc_score: float
    tokens: List[str]
    subword_tokens: List[Subword]


class ScoreParagraph(NamedTuple):
    paragraph: Paragraph
    score: float


class ParagraphClaimPersFeature(NamedTuple):
    claim_pers: PerspectiveCandidate
    feature: List[ScoreParagraph]


def select_paragraph_dp_list(ci: StaticRankedListInterface,
                             clue12_13_df,
                             tokenizer,
                             datapoint_list: List[PerspectiveCandidate]) -> List[ParagraphClaimPersFeature]:
    not_found_set = set()
    print("Load term stat")

    def subword_tokenize(word: str) -> List[Subword]:
        word = tokenizer.basic_tokenizer.clean_text(word)
        word = word.lower()
        word = tokenizer.basic_tokenizer.run_strip_accents(word)
        subwords = tokenizer.wordpiece_tokenizer.tokenize(word)
        return subwords

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

    def enum_paragraph(doc: GalagoDocRankEntry) -> Iterable[Paragraph]:
        # load tokens and BERT subword tokens
        tokens = load(TokenizedCluewebDoc, doc.doc_id)
        subword_tokens: List[List[Subword]] = lmap(subword_tokenize, tokens)
        step_size = 100
        subword_len = 350
        cursor = 0

        # return maximum index where number of subword tokens in subword_tokens[start:index] does not exist max_len
        def move_cursor(subword_tokens: List[List[Subword]], start: int, max_len: int):
            cursor_ed = start
            num_subword = 0

            def can_add_subwords():
                if cursor_ed < len(subword_tokens):
                    return num_subword + len(subword_tokens[cursor_ed]) <= max_len
                else:
                    return False

            while can_add_subwords():
                num_subword += len(subword_tokens[cursor_ed])
                cursor_ed += 1

            return cursor_ed

        while cursor < len(subword_tokens):
            cursor_ed = move_cursor(subword_tokens, cursor, subword_len)
            yield Paragraph(doc_id=doc.doc_id, doc_rank=doc.rank, doc_score=doc.score,
                            subword_tokens=list(flatten(subword_tokens[cursor:cursor_ed])),
                            tokens=tokens[cursor:cursor_ed])
            cursor += step_size


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
            paragraph_list = enum_paragraph(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            score_paragraph.sort(key=lambda p: p.score, reverse=True)
            return score_paragraph[:1]

        def get_all_paragraph_from_doc(doc: GalagoDocRankEntry) -> List[ScoreParagraph]:
            paragraph_list = enum_paragraph(doc)
            score_paragraph = lmap(paragraph_scorer_local, paragraph_list)
            return score_paragraph

        if option:
            get_paragraphs = get_best_paragraph_from_doc
        else:
            get_paragraphs = get_all_paragraph_from_doc

        candidate_paragraph: Iterable[ScoreParagraph] = flatten(lmap(get_paragraphs, ranked_docs))
        return ParagraphClaimPersFeature(claim_pers=x, feature=list(candidate_paragraph))

    r = lmap(select_paragraph_from_datapoint, datapoint_list)
    return r
