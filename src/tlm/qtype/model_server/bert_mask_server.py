import port_info
from rpc.bert_like_server import BertLikeServer
from tlm.qtype.partial_relevance.attention_based.bert_mask_predictor import get_bert_mask_predictor
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import deserialize_tuple_d


def run_server():
    predictor = get_bert_mask_predictor()

    def predict(payload):
        payload = deserialize_tuple_d(payload)
        sout = predictor.predict(payload)
        return sout

    server = BertLikeServer(predict)
    print("server started")
    server.start(port_info.BERT_MASK_PORT)



if __name__ == "__main__":
    run_server()