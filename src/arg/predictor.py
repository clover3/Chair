import cpath
from data_generator.argmining.ukp import BertDataLoader
from models.transformer import hyperparams
from models.transformer.tranformer_nli import transformer_nli
from trainer.tf_module import *
from trainer.tf_train_module import init_session


class Predictor:
    def __init__(self, topic, cheat = False, cheat_topic=None):
        self.voca_size = 30522
        self.topic = topic
        load_names = ['bert', "cls_dense"]
        if not cheat:
            run_name = "arg_key_neccesary_{}".format(topic)
        else:
            run_name = "arg_key_neccesary_{}".format(cheat_topic)
        self.hp = hyperparams.HPBert()
        self.model_dir = cpath.common_model_dir_root
        self.data_loader = BertDataLoader(topic, True, self.hp.seq_max, "bert_voca.txt")

        self.task = transformer_nli(self.hp, self.voca_size, 0, False)
        self.sess = init_session()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.load_model_white(run_name, load_names)

        self.batch_size = 512

    def encode_instance(self, topic, sentence):
        topic_str = topic + " is neccesary."
        entry = self.data_loader.encode_pair(topic_str, sentence)
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

    def predict(self, target_topic, sents):
        inputs = list([self.encode_instance(target_topic, s) for s in sents])

        def batch2feed_dict(batch):
            x0, x1, x2  = batch
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

        logits = forward_run(inputs)
        pred = np.argmax(logits, axis=1)
        return pred

    def load_model_white(self, name, include_namespace, verbose=True):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
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
        variables_to_restore = [v for v in variables if condition(v)]
        if verbose:
            print("Restoring: {} {}".format(name, id))
            for v in variables_to_restore:
                print(v)

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)
