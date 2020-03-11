from arg.claim_building.count_ngram import merge_subword
from tlm.ukp.sydney_data import dev_pretend_ukp_load_tokens_for_topic


def display():
    topic = "abortion"
    token_dict = dev_pretend_ukp_load_tokens_for_topic(topic)

    terms = ["should"]
    for doc in token_dict.values():
        for sent in doc:
            do_print = False
            if terms[0] in sent:
                sent = merge_subword(sent)
                do_print = True
                if do_print:
                    print(" ".join(sent))
            if do_print:
                print("")

if __name__ == "__main__":
    display()