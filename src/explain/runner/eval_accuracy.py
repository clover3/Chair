import os
import sys

import tensorflow as tf

from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from misc_lib import average
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import load_bert_v2
from trainer.tf_module import get_batches_ex
from trainer.tf_train_module import init_session


def eval_nli(hparam, nli_setting, run_name, dev_batches, model_path):
    print("eval_nli :", run_name)
    task = transformer_nli_pooled(hparam, nli_setting.vocab_size, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        #load_model(sess, model_path)
        load_bert_v2(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )

            cur_batch_size = len(batch[0])
            for _ in range(cur_batch_size):
                loss_list.append(loss_val)
                acc_list.append(acc)

        return average(acc_list)

    return valid_fn()


def eval_accuracy(model_path):
    hp = hyperparams.HPSENLI2()
    hp.batch_size = 128
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    run_name = os.path.split(model_path)[-1]
    dev_batches = get_batches_ex(data_loader.get_dev_data(), hp.batch_size, 4)
    acc = eval_nli(hp, nli_setting, run_name, dev_batches, model_path)
    print(run_name, acc)

if __name__  == "__main__":
    eval_accuracy(sys.argv[1])