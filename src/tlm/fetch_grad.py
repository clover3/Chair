import tensorflow as tf
import time
from models.transformer.tranformer_nli import transformer_nli_grad
from trainer.tf_train_module import *
from trainer.tf_module import epoch_runner
from data_generator.NLI import nli
from trainer.np_modules import *
from misc_lib import *
from models.transformer.hyperparams import HPBert
from log import log as log_module
from trainer.model_saver import save_model, load_bert_v2
from data_generator.shared_setting import NLI
from google_wrap.gs_wrap import download_model_last_auto
import sys
from path import output_path
from path import model_path
from datetime import datetime




def fetch_grad(hparam, vocab_size, run_name, data_loader, model_path):
    print("fetch_grad:", run_name)
    task = transformer_nli_grad(hparam, vocab_size, 0, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    loader = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    loader.restore(sess, model_path)
    dev_batches = get_batches_ex(data_loader.get_dev_data(), hparam.batch_size, 4)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    def pred_fn():
        outputs = []
        for batch in dev_batches[:100]:
            all_layer_grads,grad_emb  = sess.run([task.all_layer_grads, task.grad_emb],
                                       feed_dict=batch2feed_dict(batch)
                                       )
            outputs.append((all_layer_grads, grad_emb))

        return outputs

    r = pred_fn()

