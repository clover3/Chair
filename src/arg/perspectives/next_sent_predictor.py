import xmlrpc
import xmlrpc.client
from typing import Dict, List, Tuple

from arg.bm25 import BM25
from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict
from cpath import pjoin, data_path
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from misc_lib import NamedNumber


def pc_predict_by_bert_next_sent(bm25_module: BM25,
                                claims,
                                top_k) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    port = 8123
    # Example usage :
    proxy = xmlrpc.client.ServerProxy('http://ingham.cs.umass.edu:{}'.format(port))

    voca_path = pjoin(data_path, "bert_voca.txt")
    encoder = EncoderUnitPlain(512, voca_path)

    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        i_claim_id = int(claim_id)
        payload = []
        p_text = perspective_getter(int(p_id))
        c_text = cid_to_text[i_claim_id]
        payload.append(encoder.encode_pair(c_text, p_text))
        r = proxy.predict(payload)
        ns_score = -float(r[0])
        #ns_score = 0
        score = bm25_module.score(c_text, p_text)
        new_score = score + ns_score * 10
        score = NamedNumber(new_score, score.name + " {}".format(ns_score))
        return score

    r = predict_interface(claims, top_k, scorer)
    return r

