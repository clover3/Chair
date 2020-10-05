from collections import Counter, OrderedDict
from typing import List, NamedTuple, Iterable, Tuple

from arg.perspectives.runner_uni.build_topic_lm import ClaimLM
from arg.perspectives.select_paragraph_claim import remove_duplicate
from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import preload_man, load
from datastore.table_names import BertTokenizedCluewebDoc
from galagos.types import GalagoDocRankEntry
from list_lib import lmap, foreach, flatten
from models.classic.lm_util import get_lm_log, subtract, smooth
from models.classic.stopword import load_stopwords_ex
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature


class Record(NamedTuple):
    claim_tokens: List[str]
    doc_tokens: List[str]
    scores: List[float]
    valid_mask: List[int]


def enum_paragraph_functor(step_size, window_size):
    def enum_paragraph(docs: List[List[str]]) -> Iterable[List[str]]:

        for doc in remove_duplicate(docs):
            st = 0
            while st < len(doc):
                para: List[str] = doc[st:st + window_size]
                yield para
                st += step_size
    return enum_paragraph


class WordOnSubwords(NamedTuple):
    st: int
    ed: int
    subwords : List[str]
    word : str


def get_word_mapping(tokens):
    grouped_subwords = []
    cur_word = []
    for t in tokens:
        if t[:2] == "##":
            cur_word.append(t)
        else:
            grouped_subwords.append(cur_word)
            cur_word = [t]
    if cur_word:
        grouped_subwords.append(cur_word)


    idx = 0
    mapping = {}
    for group in grouped_subwords:
        pure_words = group[:1] + [t[2:] for t in group[1:]]
        word = "".join(pure_words)

        word_on_subwords = WordOnSubwords(idx, idx + len(group), group, word)
        for j in range(idx, idx + len(group)):
            mapping[j] = word_on_subwords
        idx += len(group)

    return mapping


def get_target_labels(tokens: List[str], log_odd, stopwords, fail_logger: Counter)\
        -> Tuple[List[float], List[int]]:
    mapping = get_word_mapping(tokens)

    scores = []
    masks = []
    for idx, t in enumerate(tokens):
        word_on_subwords = mapping[idx]
        word = word_on_subwords.word

        per_token_score = 0
        if word in log_odd:
            if word not in stopwords:
                raw_score = log_odd[word]
                per_token_score = raw_score / len(word_on_subwords.subwords)
                mask = 1
            else:
                mask = 0
        else:
            fail_logger[word] += 1
            mask = 0

        masks.append(mask)
        scores.append(per_token_score)

    return scores, masks


def get_generator(max_seq_length, bg_lm, alpha):
    log_bg_lm = get_lm_log(bg_lm)
    top_n = 100
    stopwords = load_stopwords_ex()
    fail_logger = Counter()
    bert_tokenizer = get_tokenizer()

    def generate(claim_lm: ClaimLM,
                 ranked_list: List[GalagoDocRankEntry]) -> List[Record]:
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

        def get_record(tokens) -> Record:
            scores, masks = get_target_labels(tokens, log_odd, stopwords, fail_logger)
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


def write_records(records: List[Record],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def encode(record: Record) -> OrderedDict:
        tokens = ["[CLS]"] + record.claim_tokens + ["[SEP]"] + record.doc_tokens + ["[SEP]"]
        segment_ids = [0] * (len(record.claim_tokens) + 2) \
                      + [1] * (len(record.doc_tokens) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer,
                                           max_seq_length,
                                           tokens,
                                           segment_ids)

        labels = [0.] * (len(record.claim_tokens) + 2) + record.scores
        labels += (max_seq_length - len(labels)) * [0.]
        label_mask = [0] * (len(record.claim_tokens) + 2) + record.valid_mask
        label_mask += (max_seq_length - len(label_mask)) * [0]
        features['label_ids'] = create_float_feature(labels)
        features['label_masks'] = create_int_feature(label_mask)
        return features

    writer = RecordWriterWrap(output_path)
    features: List[OrderedDict] = lmap(encode, records)
    foreach(writer.write_feature, features)
    writer.close()


