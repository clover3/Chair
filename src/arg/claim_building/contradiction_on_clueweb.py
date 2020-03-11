import sys
import xmlrpc.client

from arg.claim_building.contradiction_predictor import PORT_CONFLICT_EX
from cache import load_from_pickle, save_to_pickle, StreamPickler
from data_generator.common import get_encoder_unit
from misc_lib import assign_list_if_not_exists, pick2
from models.classic.stopword import load_stopwords
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic


def build_inv_index(sents):
    stopword = load_stopwords()
    stopword.add("should")
    group = {}

    for idx, sent in enumerate(sents):
        for t in sent:
            if len(t) > 1 and t not in stopword:
                assign_list_if_not_exists(group, t)
                group[t].append(idx)
    return group


def pair_generator():
    topic = "abortion"
    sents = load_from_pickle(get_sent_pickle_name(topic))
    print("building inv-index")
    inv_index = build_inv_index(sents)

    len_info = list([(key, len(inv_index[key])) for key in inv_index])
    len_info.sort(key=lambda x:x[1], reverse=True)

    sent_per_cluster = 10000
    for key, _ in len_info:
        sent_indice = inv_index[key]
        loop = min(sent_per_cluster, len(sent_indice)*len(sent_indice))
        for _ in range(loop):
            idx1, idx2 = pick2(sent_indice)
            sent1 = sents[idx1]
            sent2 = sents[idx2]

            yield (sent1, sent2)


def get_sent_pickle_name(topic):
    return "should_sent_{}".format(topic)


def extract_should_sent():
    # select sentences that has should in it
    topic = "abortion"
    token_dict = ukp_load_tokens_for_topic(topic)
#    token_dict = dev_pretend_ukp_load_tokens_for_topic(topic)

    should_sents = []
    for doc in token_dict.values():
        for sent in doc:
            if "should" in sent:
                should_sents.append(sent)
                #print(sent)

    save_to_pickle(should_sents, get_sent_pickle_name(topic))


def run_predictions():
    pair_gen = pair_generator()
    max_sequence = 300
    encoder_unit = get_encoder_unit(max_sequence)
    proxy = xmlrpc.client.ServerProxy('http://ingham.cs.umass.edu:{}'.format(PORT_CONFLICT_EX))
    batch_size = 16 * 10
    sp = StreamPickler("contradiction_prediction", 100 * 1000)
    cont_cnt = 0
    all_cnt = 0
    try:
        while True:
            payload = []
            for _ in range(batch_size):
                sent1, sent2 = pair_gen.__next__()
                d = encoder_unit.encode_token_pairs(sent1, sent2)
                e = d['input_ids'], d['input_mask'], d['segment_ids']
                payload.append(e)
            r = proxy.predict(payload)
            assert len(r) == len(payload)

            for e, p in zip(payload, r):
                logits, _ = p
                if logits[2] > logits[1] and logits[2] > logits[0]:
                    sp.add((e, p))
                    cont_cnt += 1
            all_cnt += batch_size

            if all_cnt == batch_size * 100:
                print(cont_cnt, all_cnt)
                break
    except StopIteration:
        pass
    sp.flush()
    ##

if __name__ == "__main__":
    command = sys.argv[1]
    locals()[command]()
