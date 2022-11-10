
import sys
import traceback

from rpc.bert_like_server import RPCServerWrap
from tlm.next_sent.preditor import Predictor

PORT = 8123


def run_server(model_path):
    predictor = Predictor(model_path)
    # payload is list of (input_ids, input_mask, segment_ids)
    def predict(payload):
        try:
            sout = predictor.predict(payload)
            return sout
        except Exception as e:
            print("Exception in user code:")
            print(traceback.print_exc(file=sys.stdout))
            print(e)
        return []

    server = RPCServerWrap(predict)
    print("server started")
    server.start(PORT)


if __name__ == "__main__":
    run_server(sys.argv[1])