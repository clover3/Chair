import port_info
from rpc.bert_like_server import RPCServerWrap
from bert_api.bert_masking_common import deserialize_tuple_d
from bert_api.task_clients.nli_interface.nli_mask_predictor import get_nli_bert_mask_predictor
import sys

def run_server():
    model_path = sys.argv[1]
    predictor = get_nli_bert_mask_predictor(model_path)

    def predict(payload):
        payload = deserialize_tuple_d(payload)
        sout = predictor.predict(payload)
        return sout

    server = RPCServerWrap(predict)
    print("server started")
    server.start(port_info.BERT_MASK_PORT)



if __name__ == "__main__":
    run_server()