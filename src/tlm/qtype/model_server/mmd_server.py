import traceback

import cpath
import port_info
import tlm.model.base as bert
from cpath import output_path
from models.transformer import hyperparams
from rpc.bert_like_server import RPCServerWrap
from tf_v2_support import disable_eager_execution
from tf_v2_support import placeholder
from trainer.tf_module import *
from trainer.tf_train_module_v2 import init_session


class transformer_logit:
    def __init__(self, hp, num_classes, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = placeholder(tf.int64, [None, seq_length])
        input_mask = placeholder(tf.int64, [None, seq_length])
        segment_ids = placeholder(tf.int64, [None, seq_length])
        label_ids = placeholder(tf.int64, [None])
        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pooled_output = self.model.get_pooled_output()
        logits = tf.keras.layers.Dense(num_classes, name="cls_dense")(pooled_output)
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)
        loss = tf.reduce_mean(input_tensor=loss_arr)

        self.loss = loss
        self.logits = logits

    def batch2feed_dict(self, batch):
        if len(batch) == 3:
            x0, x1, x2 = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
            }
        else:
            x0, x1, x2, y = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
                self.y: y,
            }
        return feed_dict


class PredictorClsDense:
    def __init__(self, num_classes, seq_len=None):
        disable_eager_execution()
        self.voca_size = 30522
        self.hp = hyperparams.HPFAD()
        if seq_len is not None:
            self.hp.seq_max = seq_len
        self.model_dir = cpath.common_model_dir_root
        self.task = transformer_logit(self.hp, num_classes, self.voca_size, False)
        self.sess = init_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.batch_size = 64

    def predict(self, triple_list):
        try:
            return self._predict(triple_list)
        except Exception as e:
            traceback.print_exc()
            raise

    def _predict(self, triple_list):
        print("Processing {} items".format(len(triple_list)))
        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                self.task.x_list[0]: x0,
                self.task.x_list[1]: x1,
                self.task.x_list[2]: x2,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.batch_size, 3)
            logit_list = []
            for batch in batches:
                logits,  = self.sess.run([self.task.logits, ],
                                         feed_dict=batch2feed_dict(batch))
                logit_list.append(logits)
            return np.concatenate(logit_list)

        scores = forward_run(triple_list)
        return scores.tolist()

    def load_model_white(self, save_dir, include_namespace, verbose=True):
        def get_last_id(save_dir):
            last_model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]
                        if last_model_id is None:
                            last_model_id = model_id
                        else:
                            last_model_id = model_id if model_id > last_model_id else last_model_id
            return last_model_id

        id = get_last_id(save_dir)
        path = os.path.join(save_dir, "{}".format(id))

        self.loader = tf.compat.v1.train.Saver(max_to_keep=1)
        self.loader.restore(self.sess, path)



def run_server():
    save_path = os.path.join(output_path, "model", "runs", "mmd_2M")

    disable_eager_execution()

    predictor = PredictorClsDense(1, 512)
    load_names = ['bert', "cls_dense"]
    predictor.load_model_white(save_path, load_names)

    def predict(payload):
        sout = predictor.predict(payload)
        return sout

    server = RPCServerWrap(predict)
    print("server started")
    server.start(port_info.MMD_PORT)


if __name__ == "__main__":
    run_server()