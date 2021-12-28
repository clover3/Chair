from cpath import output_path
from port_info import MMD_Z_PORT
from rpc.bert_like_server import BertLikeServer
from tf_v2_support import disable_eager_execution
from tlm.qtype.model_server.mmd_server import PredictorClsDense
from trainer.tf_module import *


def run_server():
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")

    disable_eager_execution()

    predictor = PredictorClsDense(2, 512)
    load_names = ['bert', "cls_dense"]
    predictor.load_model_white(save_path, load_names)

    def predict(payload):
        sout = predictor.predict(payload)
        return sout

    server = BertLikeServer(predict)
    print("server started")
    server.start(MMD_Z_PORT)


if __name__ == "__main__":
    run_server()