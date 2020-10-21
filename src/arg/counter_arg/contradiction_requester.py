from bert_api.client_lib import BERTClient
from trainer.promise import PromiseKeeper, MyPromise


class Request:
    def __init__(self):
        self.client = BERTClient("http://localhost", 8122, 300)

    def request_multiple(self, payload):
        pk = PromiseKeeper(self.client.request_multiple)

        future_list = []
        for sent1, sent2 in payload:
            if sent1.strip() and sent2.strip():
                p = MyPromise((sent1, sent2), pk)
                future_list.append(p.future())
        pk.do_duty()

        r = []
        for (sent1, sent2), f in zip(payload, future_list):
            probs, token_scores = f.get()
            r.append((sent1, sent2, probs))
        return r


class RequestEx:
    def __init__(self):
        self.client = BERTClient("http://localhost", 8122, 300)

    def request_multiple(self, payload):
        pk = PromiseKeeper(self.client.request_multiple)

        future_list = []
        for sent1, sent2 in payload:
            if sent1.strip() and sent2.strip():
                p = MyPromise((sent1, sent2), pk)
                future_list.append(p.future())
        pk.do_duty()

        r = []
        for (sent1, sent2), f in zip(payload, future_list):
            probs, token_scores = f.get()
            r.append((sent1, sent2, probs, token_scores))
        return r