



# TODO foreach of the claim
# TODO   get_relevant_unigrams_for_claim
# TODO   get ranked_list for claim
# TODO      for doc in ranked_list
# TODO         for all term,
# TODO         update tf,df of (term, controversy), (term,no controversy)
# TODO         update tf,df of (term)
# TODO
# TODO
# TODO
# TODO
import nltk

from arg.perspectives import es_helper
from arg.perspectives.clueweb_helper import ClaimRankedList, load_doc
from arg.perspectives.context_analysis_routine import count_term_stat, check_hypothesis
from arg.perspectives.load import load_dev_claim_ids, get_claims_from_ids
from misc_lib import lmap, foreach


def get_perspective(claim):
    cid = claim["cId"]
    claim_text = claim["text"]
    lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)
    perspectives = []
    for _text, _pid, _score in lucene_results:
        perspectives.append((_text, _pid, _score))
    return claim_text, perspectives



def claim_language_model_property():
    dev_claim_ids = load_dev_claim_ids()
    claims = get_claims_from_ids(dev_claim_ids)
    all_ranked_list = ClaimRankedList()
    all_voca = set()

    def load_and_format_doc(doc_id):
        try:
            tokens = load_doc(doc_id)
            print(doc_id)
        except KeyError:
            print("doc {} not found".format(doc_id))
            raise

        token_set = set(tokens)
        all_voca.update(token_set)
        return {'doc_id': doc_id,
                'tokens': tokens,
                'tokens_set': token_set,
                }

    def get_relevant_unigrams(perspectives):
        unigrams = set()
        tokens = [nltk.word_tokenize(_text) for _text, _pid, _score in perspectives]
        foreach(unigrams.update, tokens)
        return unigrams


    for claim in claims:
        claim_text, perspective = get_perspective(claim)
        unigrams = get_relevant_unigrams(perspective)
        ranked_list = all_ranked_list.get(str(claim['cId']))
        doc_ids = [t[0] for t in ranked_list]
        print("Loading documents")
        docs = lmap(load_and_format_doc, doc_ids)
        print("counting terms stat")

        count_term_stat(docs, unigrams)

        cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, \
        clueweb_df, clueweb_tf, ctf_cont, ctf_ncont, \
        df_cont, df_ncont, tf_cont, tf_ncont = count_term_stat(docs, unigrams)

        # check hypothesis
        check_hypothesis(all_voca, cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf, ctf_cont,
                         ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont, unigrams)



if __name__ == "__main__":
    claim_language_model_property()
