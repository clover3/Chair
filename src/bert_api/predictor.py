from typing import List, Tuple

import cpath
from models.transformer import hyperparams
from models.transformer.transfomer_logit import transformer_logit
from trainer.tf_module import *
from trainer.tf_train_module import init_session


class Predictor:
    def __init__(self, model_path, num_classes, max_seq_length=None):
        self.voca_size = 30522
        load_names = ['bert', "output_bias", "output_weights"]
        self.hp = hyperparams.HPFAD()
        if max_seq_length is not None:
            self.hp.seq_max = max_seq_length
        self.model_dir = cpath.model_path
        self.task = transformer_logit(self.hp, num_classes, self.voca_size, False)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.load_model_white(model_path, load_names)
        self.batch_size = 64

    def predict(self, triple_list: List[Tuple[List, List, List]]):
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
            return np.concatenate(logit_list) if logit_list else np.array([])

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

        def condition(v):
            if v.name.split('/')[0] in include_namespace:
                return True
            return False

        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = variables
        if verbose:
            for v in variables_to_restore:
                print(v)

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)
