from nltk import word_tokenize

from arg.claim_building.count_ngram import merge_subword
from arg.perspectives.context_analysis_routine import analyze
from arg.perspectives.load import get_claims_from_ids
from cache import save_to_pickle, load_from_pickle
from tlm.ukp.sydney_data import dev_pretend_ukp_load_tokens_for_topic

claim_id = 641


def get_perspective_candidates(claim_id):
    from arg.perspectives import es_helper
    claims = get_claims_from_ids([claim_id])
    claim_text = claims[0]['text']
    lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)
    for _text, _pid, _score in lucene_results:
        yield _text, _pid


school_uniform_perspective_candidates_pickle_name = "school_uniform_perspective_candidates_pickle_name"


def save_candidate_perspectives_for_claim():
    obj = list(get_perspective_candidates(claim_id))
    save_to_pickle(obj, school_uniform_perspective_candidates_pickle_name)


def load_candidate_perspectives_for_claim():
    return load_from_pickle(school_uniform_perspective_candidates_pickle_name)


def get_relevant_unigrams_for_claim():
    unigrams = set()
    for text, pid in load_candidate_perspectives_for_claim():
        tokens = word_tokenize(text.lower())
        unigrams.update(tokens)
    return unigrams


def work():
    # load tokens for school_uniform
    topic = "school_uniform"
    unigrams = get_relevant_unigrams_for_claim()
    tokens_dict = dev_pretend_ukp_load_tokens_for_topic(topic)
    print("Total of {} unigrams of interest".format(len(unigrams)))
    all_voca = set()

    def merge_subword_in_doc(doc):
        r = []
        for segment in doc:
            r += merge_subword(segment)
        return r

    def format_doc(k, v):
        tokens = merge_subword_in_doc(v)
        tokens_set = set(tokens)
        all_voca.update(tokens_set)
        return {'doc_id': k,
                'tokens': tokens,
                'tokens_set': tokens_set,
                }

    doc_list = [format_doc(k, v) for k, v in tokens_dict.items()]

    analyze(all_voca, doc_list, unigrams)


if __name__ == "__main__":
    work()
