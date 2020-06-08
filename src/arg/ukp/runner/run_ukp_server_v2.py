import sys

from bert_api.predictor_v2 import Predictor
from rpc.bert_like_server import BertLikeServer
from tf_v2_support import disable_eager_execution

PORT_UKP = 8123

def run_server(model_path):
    disable_eager_execution()

    predictor = Predictor(model_path, 3, 300)
    def predict(payload):
        sout = predictor.predict(payload)
        return sout

    server = BertLikeServer(predict)
    print("server started")
    server.start(PORT_UKP)


if __name__ == "__main__":
    run_server(sys.argv[1])