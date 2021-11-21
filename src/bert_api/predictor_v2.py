import cpath
from models.transformer import hyperparams
from models.transformer.transformer_logit_v2 import transformer_logit
from tf_v2_support import disable_eager_execution
from trainer.tf_module import *
from trainer.tf_train_module_v2 import init_session


class Predictor:
    def __init__(self, model_path, num_classes, seq_len=None):
        disable_eager_execution()
        self.voca_size = 30522
        load_names = ['bert', "output_bias", "output_weights"]
        self.hp = hyperparams.HPFAD()
        if seq_len is not None:
            self.hp.seq_max = seq_len
        self.model_dir = cpath.model_path
        self.task = transformer_logit(self.hp, num_classes, self.voca_size, False)
        self.sess = init_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.load_model_white(model_path, load_names)
        self.batch_size = 64

    def predict(self, triple_list):
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
                logits,  = self.sess.run([self.task.sout, ],
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


