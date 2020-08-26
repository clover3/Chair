from collections import Counter
from typing import List, Iterable

from arg.perspectives.claim_lm.token_score_datagen import enum_paragraph_functor, get_target_labels
from arg.perspectives.runner_uni.build_topic_lm import get_lm_log, ClaimLM, smooth, subtract
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import preload_man
from datastore.table_names import BertTokenizedCluewebDoc
from galagos.types import GalagoDocRankEntry
from list_lib import lmap
from models.classic.stopword import load_stopwords_ex


def get_generator(max_seq_length, bg_lm, alpha):
    log_bg_lm = get_lm_log(bg_lm)
    top_n = 100
    stopwords = load_stopwords_ex()
    fail_logger = Counter()
    bert_tokenizer = get_tokenizer()

    def generate(claim_lm: ClaimLM,
                 ranked_list: List[GalagoDocRankEntry]):
        claim_text = claim_lm.claim
        claim_tokens = bert_tokenizer.tokenize(claim_text)
        claim_token_len = len(claim_tokens)

        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)
        doc_ids = lmap(lambda x: x.doc_id, ranked_list[:top_n])
        print("loading docs")
        preload_man.preload(BertTokenizedCluewebDoc, doc_ids)

        window_size = max_seq_length - claim_token_len - 3
        step_size = max_seq_length - 112
        enum_paragraph = enum_paragraph_functor(step_size, window_size)

        def get_record(tokens):
            scores, masks = get_target_labels(tokens, log_odd, stopwords, fail_logger)
            sum(scores)
            return Record(claim_tokens, tokens, scores, masks)

        tokens_list: List[List[str]] = []
        not_found = 0
        for doc_id in doc_ids:
            try:
                tokens: List[str] = list(flatten(load(BertTokenizedCluewebDoc, doc_id)))
                tokens_list.append(tokens)
            except KeyError:
                not_found += 1
                pass

        print("{} of {} not found".format(not_found, len(tokens_list)))
        paragraph_list: Iterable[List[str]] = enum_paragraph(tokens_list)
        records: List[Record] = lmap(get_record, paragraph_list)

        return records

    return generate