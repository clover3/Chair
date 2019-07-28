import random
from cache import *
from misc_lib import *
from collections import Counter
from data_generator import tokenizer_b as tokenization
from models.classic.stopword import load_stopwords
import path
from tlm.stem import CacheStemmer
from tlm.retrieve_doc import RobustCollection

def translate_mask2token_level(sent_list, target_tokens, mask_indice, tokenizer):
    basic_tokens = list([tokenizer.basic_tokenizer.tokenize(s) for s in sent_list])
    sub_tokens_tree = []
    for sent_tokens in basic_tokens:
        sub_tokens_tree.append(list())
        for token in sent_tokens:
            r = tokenizer.wordpiece_tokenizer.tokenize(token)
            sub_tokens_tree[-1].append(r)

    # Iter[List[tokens]]

    sent_idx = 0
    token_idx = 0
    local_subword_idx = 0
    global_subword_idx = 0
    mask_indice.sort()
    word_mask_indice = []
    mask_indice_idx = 0

    while sent_idx < len(basic_tokens) and mask_indice_idx < len(mask_indice):
        st = sub_tokens_tree[sent_idx][token_idx][local_subword_idx]

        if tokenizer.basic_tokenizer.do_lower_case:
            assert target_tokens[global_subword_idx] == st

        if global_subword_idx == mask_indice[mask_indice_idx]:
            word_mask_indice.append((sent_idx, token_idx))
            mask_indice_idx += 1


        local_subword_idx += 1
        global_subword_idx += 1

        if local_subword_idx == len(sub_tokens_tree[sent_idx][token_idx]):
            token_idx += 1
            local_subword_idx = 0

        if token_idx == len(basic_tokens[sent_idx]):
            sent_idx += 1
            token_idx = 0

    return basic_tokens, word_mask_indice


def remove(tokens_list, indice):
    r = []
    for sent_id, sent in enumerate(tokens_list):
        for token_idx, token in enumerate(sent):
            if not (sent_id, token_idx) in indice:
                r.append(token)
    return r


def get_visible(sent_list, target_tokens, mask_indice, tokenizer):
    basic_tokens, word_mask_indice = translate_mask2token_level(sent_list, target_tokens, mask_indice, tokenizer)
    return remove(basic_tokens, word_mask_indice)


class HintRetriever:
    def __init__(self):
        self.stopword = load_stopwords()
        self.stemmer = CacheStemmer()
        vocab_file = os.path.join(path.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        tprint("Loading inv_index for robust")
        self.collection = RobustCollection()
        tprint("Done")
        self.num_candidate = 10

    def retrieve_hints(self, inst):
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice = inst
        basic_tokens, word_mask_indice = translate_mask2token_level(sent_list, target_tokens, mask_indice, self.tokenizer)
        # stemming

        visible_tokens = remove(basic_tokens, word_mask_indice)
        stemmed = list([self.stemmer.stem(t) for t in visible_tokens])
        query = self.filter_stopword(stemmed)
        res = self.collection.retrieve_docs(query)

        candi_list = left(res[1:1+self.num_candidate])

        r = target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, candi_list
        return r

    def query_gen(self, inst):
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = inst
        basic_tokens, word_mask_indice = translate_mask2token_level(sent_list, target_tokens, mask_indice, self.tokenizer)
        # stemming

        visible_tokens = remove(basic_tokens, word_mask_indice)
        stemmed = list([self.stemmer.stem(t) for t in visible_tokens])
        query = self.filter_stopword(stemmed)
        high_qt_stmed = self.collection.high_idf_q_terms(Counter(query))

        final_query = []
        for t in visible_tokens:
            if self.stemmer.stem(t) in high_qt_stmed:
                final_query.append(t)
        return final_query


    def filter_stopword(self, l):
        return list([t for t in l if t not in self.stopword])




def main():
    hr = HintRetriever()

    spr = StreamPickleReader("robust_problems_")
    sp_candi_q = StreamPickler("robust_candi_query_", 1000)
    sp_prob_q = StreamPickler("robust_problem_q_", 1000)

    ticker = TimeEstimator(1000 * 1000)
    qid = 0

    while spr.has_next():
        inst = spr.get_item()
        query = hr.query_gen(inst)

        inst_q = (inst, qid)
        sp_prob_q.add(inst_q)
        sp_candi_q.add((qid, query))
        qid += 1
        ticker.tick()
    sp_prob_q.flush()
    sp_candi_q.flush()


if __name__ == "__main__":
    main()
