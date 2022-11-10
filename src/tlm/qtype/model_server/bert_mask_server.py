import port_info
from rpc.bert_like_server import RPCServerWrap
from bert_api.bert_masking_common import deserialize_tuple_d
from bert_api.task_clients.mmd_z_interface.mmd_z_mask_predictor import get_mmd_z_bert_mask_predictor


def run_server():
    predictor = get_mmd_z_bert_mask_predictor()

    def predict(payload):
        payload = deserialize_tuple_d(payload)
        sout = predictor.predict(payload)
        return sout

    server = RPCServerWrap(predict)
    print("server started")
    server.start(port_info.BERT_MASK_PORT)


 
if __name__ == "__main__":
    run_server()