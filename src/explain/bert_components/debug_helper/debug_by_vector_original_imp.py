import sys

import tensorflow as tf

from data_generator.shared_setting import BertNLI
from explain.bert_components.debug_helper.misc_debug_common import load_data, evaluate_acc_for_batches
from models.transformer import hyperparams
from models.transformer.transformer_cls import transformer_pooled
from trainer.model_saver import load_model
from trainer.tf_train_module import init_session


def show_vector():
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    task = transformer_pooled(hp, nli_setting.vocab_size, False)
    # first_token_tensor = tf.squeeze(last_layer_out[:, 0:1, :], axis=1)
    sess = init_session()
    save_path = sys.argv[1]
    sess.run(tf.global_variables_initializer())
    load_model(sess, save_path)
    dev_batches = load_data(300, 2)
    x0, x1, x2, y = dev_batches[0]
    out_value, = sess.run([task.logits],
             feed_dict={
                 task.x_list[0]: x0,
                 task.x_list[1]: x1,
                 task.x_list[2]: x2,
             })
    print(list(out_value[0]))


def show_acc():
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    task = transformer_pooled(hp, nli_setting.vocab_size, False)
    sess = init_session()
    save_path = sys.argv[1]
    sess.run(tf.global_variables_initializer())
    load_model(sess, save_path)
    dev_batches = load_data(300, 8)

    def batch_predict(batch):
        x0, x1, x2, y = batch
        logits, = sess.run([task.logits],
                           feed_dict={
                               task.x_list[0]: x0,
                               task.x_list[1]: x1,
                               task.x_list[2]: x2,
                           })
        return logits, y

    acc = evaluate_acc_for_batches(batch_predict, dev_batches)

    print("acc", acc)



if __name__ == "__main__":
    show_acc()
