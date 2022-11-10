import sys

from bert_api.predictor import Predictor
from rpc.bert_like_server import RPCServerWrap

PORT_CONFLICT_EX = 8122


def run_server(model_path):
    predictor = Predictor(model_path, 3, 300)
    # payload is list of (input_ids, input_mask, segment_ids)
    def predict(payload):
        sout = predictor.predict(payload)
        return sout

    server = RPCServerWrap(predict)
    print("server started")
    server.start(PORT_CONFLICT_EX)


if __name__ == "__main__":
    run_server(sys.argv[1])