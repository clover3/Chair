
import cpath
import port_info
from cpath import output_path
from rpc.bert_like_server import RPCServerWrap
from tf_v2_support import disable_eager_execution
from tlm.model_cnfig import JsonConfig
from tlm.qtype.model_server.qde_server import Predictor
from trainer.tf_module import *


class ModelConfig:
    max_seq_length = 512
    q_type_voca = 2048
    q_max_seq_length = 128

def run_server():
    save_path = os.path.join(output_path, "model", "runs", "qtype_2Y", "model.ckpt-120000")
    disable_eager_execution()
    bert_config_file = os.path.join(cpath.data_path, "bert_config.json")
    config = JsonConfig.from_json_file(bert_config_file)
    model_config = ModelConfig()
    config.set_attrib("q_voca_size", model_config.q_type_voca)
    config.set_attrib("max_seq_length", model_config.max_seq_length)
    config.set_attrib('q_max_seq_length', model_config.q_max_seq_length)
    predictor = Predictor(config)
    loader = tf.compat.v1.train.Saver(max_to_keep=1)
    loader.restore(predictor.sess, save_path)

    def predict(payload):
        return predictor.predict(payload)

    server = RPCServerWrap(predict)
    print("server started with port {}".format(port_info.QDE_PORT))
    server.start(port_info.QDE_PORT)


if __name__ == "__main__":
    run_server()
