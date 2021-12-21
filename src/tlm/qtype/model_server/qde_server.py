import cpath
from cpath import output_path
from models.transformer.bert_common_v2 import create_initializer
from rpc.bert_like_server import BertLikeServer
from tf_v2_support import disable_eager_execution
from tf_v2_support import placeholder
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model.get_hidden_v2 import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qde_model_fn import qtype_modeling_single_mlp
from trainer.tf_module import *
from trainer.tf_train_module_v2 import init_session


class qde_model:
    def __init__(self, config, is_training=True):
        seq_length = config.max_seq_length
        q_seq_length = config.q_max_seq_length
        qe_input_ids = placeholder(tf.int64, [None, q_seq_length])
        qe_input_mask = placeholder(tf.int64, [None, q_seq_length])
        qe_segment_ids = placeholder(tf.int64, [None, q_seq_length])
        de_input_ids = placeholder(tf.int64, [None, seq_length])
        de_input_mask = placeholder(tf.int64, [None, seq_length])
        de_segment_ids = placeholder(tf.int64, [None, seq_length])

        label_ids = placeholder(tf.int64, [None])
        self.x_list = [qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids]
        self.y = label_ids
        use_one_hot_embeddings = True
        def single_bias_model(config, vector):
            dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                          kernel_initializer=create_initializer(config.initializer_range))
            v = dense(vector)
            return tf.reshape(v, [-1])

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=qe_input_ids,
                input_mask=qe_input_mask,
                token_type_ids=qe_segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output()  # [batch_size * 2, hidden_size]
            qtype_vector1 = qtype_modeling_single_mlp(config, pooled1)  # [batch_size * 2, qtype_length]
            q_bias = single_bias_model(config, pooled1)
        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=de_input_ids,
                input_mask=de_input_mask,
                token_type_ids=de_segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = qtype_modeling_single_mlp(config, pooled2)
            d_bias = single_bias_model(config, pooled2)
        self.qtype_vector1 = qtype_vector1
        self.qtype_vector2 = qtype_vector2

        query_document_score1 = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score2 = query_document_score1 + bias
        query_document_score3 = query_document_score2 + q_bias
        query_document_score = query_document_score3 + d_bias
        self.logits = query_document_score
        self.d_bias = d_bias
        self.q_bias = q_bias


class Predictor:
    def __init__(self, config):
        disable_eager_execution()
        self.voca_size = 30522
        self.model_dir = cpath.model_path
        self.task = qde_model(config, False)
        self.sess = init_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.batch_size = 64

    def predict(self, triple_list):
        def batch2feed_dict(batch):
            x_list = list(batch)
            feed_dict = {}
            for tensor, value in zip(self.task.x_list, x_list):
                feed_dict[tensor] = value
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.batch_size, 6)
            tensor_d = {
                'logits': self.task.logits,
                'qtype_vector1': self.task.qtype_vector1,
                'qtype_vector2': self.task.qtype_vector2,
                'd_bias': self.task.d_bias,
                'q_bias': self.task.q_bias,
            }
            keys = list(tensor_d.keys())
            tensors = [tensor_d[k] for k in keys]
            value_d_list = defaultdict(list)
            for batch in batches:
                values = self.sess.run(tensors,
                                         feed_dict=batch2feed_dict(batch))
                for k, value in zip(keys, values):
                    value_d_list[k].append(value)

            temp_d = {}
            for k, value_list in value_d_list.items():
                value_all = np.concatenate(value_list)
                num_items = len(value_all)
                temp_d[k] = value_all.tolist()

            output = []
            for j in range(num_items):
                cur_d = {}
                for k in keys:
                    cur_d[k] = temp_d[k][j]
                output.append(cur_d)
            return output

        scores = forward_run(triple_list)
        return scores


class ModelConfig:
    max_seq_length = 512
    q_type_voca = 2048
    q_max_seq_length = 128


def run_server():
    save_path = os.path.join(output_path, "model", "runs", "qtype_2T", "model.ckpt-200000")
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

    server = BertLikeServer(predict)
    print("server started")
    server.start(8128)


if __name__ == "__main__":
    run_server()
