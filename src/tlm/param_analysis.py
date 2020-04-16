from models.transformer.tranformer_nli import transformer_nli_grad, transformer_nli_hidden
from trainer.model_saver import load_model_w_scope
from trainer.np_modules import *
from trainer.tf_train_module import *


def fetch_hidden_vector(hparam, vocab_size, run_name, data_loader, model_path):
    print("fetch_hidden_vector:", run_name)
    task = transformer_nli_hidden(hparam, vocab_size, 0, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    load_model_w_scope(sess, model_path, ["bert"])
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
            x0, x1, x2, y = batch
            all_layers, emb_outputs = sess.run([vars],
                                   feed_dict=batch2feed_dict(batch)
                                   )
            outputs.append((all_layers, emb_outputs, x0))

        return outputs

    return pred_fn()


def fetch_params(hparam, vocab_size, run_name, data_loader, model_path):
    print("fetch_hidden_vector:", run_name)
    task = transformer_nli_hidden(hparam, vocab_size, 0, False)
    sess = init_session()
    sess.run(tf.global_variables_initializer())

    load_model_w_scope(sess, model_path, ["bert"])
    vars = tf.trainable_variables()
    names = list([v.name for v in vars])

    vars_out, = sess.run([vars])
    return names, vars_out


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
        grads = []
        logit_list = []
        target = []
        for i in range(len(task.all_layer_grads)):
            print(task.all_layer_grads[i])
            target.append(task.all_layer_grads[i])

        target.append(task.grad_emb)
        for batch in dev_batches[:100]:
            r, logits = sess.run([target, task.logits],
                                   feed_dict=batch2feed_dict(batch)
                                   )
            grads.append(r)
            logit_list.append(logits)

        return grads, logit_list

    return pred_fn()

