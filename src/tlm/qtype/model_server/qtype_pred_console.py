from typing import Dict

import numpy as np
from scipy.special import softmax

from bert_api.client_lib_msmarco import BERTClientMSMarco
from cache import load_from_pickle


def main():
    client = BERTClientMSMarco("http://localhost", 8123, 512)
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    qtype_id_mapping_rev = {v:k for k, v in qtype_id_mapping.items()}
    qtype_id_mapping_rev[0] = "OOV"
    while True:
        sent1 = input("Query: ")
        ret = client.request_single(sent1, "")
        probs_arr = softmax(ret, axis=1)
        probs = probs_arr[0]
        rank = np.argsort(probs)[::-1]
        for idx in rank[:100]:
            print(idx, qtype_id_mapping_rev[idx], probs[idx])
        print()


if __name__ == "__main__":
    main()
