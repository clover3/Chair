from datetime import datetime

from cpath import output_path
from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from log import log as log_module
from misc_lib import *
from models.transformer.hyperparams import HPBert
from models.transformer.tranformer_nli import transformer_nli
from tf_v2_support import disable_eager_execution
from tlm.dictionary.nli_w_dict import setup_summary_writer
from trainer.model_saver import save_model, load_bert_v2, get_model_path
from trainer.tf_module import epoch_runner, get_nli_batches_from_data_loader
from trainer.tf_train_module import *


def train_nli(hparam, nli_setting, run_name, num_epochs, data, model_path):
    print("Train nil :", run_name)
    task = transformer_nli(hparam, nli_setting.vocab_size, 2)
    with tf.variable_scope("optimizer"):
        train_cls, global_step = get_train_op(task.loss, hparam.lr)

    train_batches, dev_batches = data

    log = log_module.train_logger()

    def get_summary_obj(loss, acc):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag='loss', simple_value=loss)
        summary.value.add(tag='accuracy', simple_value=acc)
        return summary

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    train_writer, test_writer = setup_summary_writer(run_name)
    if model_path is not None:
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

    g_step_i = 0
    def train_classification(batch, step_i):
        loss_val, acc,  _ = sess.run([task.loss, task.acc, train_cls,
                                                    ],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
        log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        train_writer.add_summary(get_summary_obj(loss_val, acc), step_i)
        g_step_i = step_i
        return loss_val, acc


    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches[:100]:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        loss_val = average(loss_list)
        acc = average(acc_list)
        train_writer.add_summary(get_summary_obj(loss_val, acc), g_step_i)

        return average(acc_list)

    def save_fn():
        return save_model(sess, run_name, global_step)

    print("{} train batches".format(len(train_batches)))
    valid_freq = 100
    save_interval = 100000
    for i_epoch in range(num_epochs):
        loss, _ = epoch_runner(train_batches, train_classification,
                               valid_fn, valid_freq,
                               save_fn, save_interval)


    return save_fn()

def test_nli(hparam, nli_setting, run_name, data, model_path):
    print("test nil :", run_name)
    task = transformer_nli(hparam, nli_setting.vocab_size, 2, False)

    train_batches, dev_batches = data


    sess = init_session()
    sess.run(tf.global_variables_initializer())

    loader = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    loader.restore(sess, model_path)

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
        for batch in dev_batches[:100]:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        return average(acc_list)

    return valid_fn()


def save_report(task, run_name, init_model, avg_acc):
    file_name = "{}_{}".format(run_name, init_model)
    p = os.path.join(output_path, "report", file_name)
    exist_or_mkdir(os.path.join(output_path, "report"))
    f = open(p, "w")
    time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    f.write("{}\n".format(time_str))
    f.write("{}\t{}\n".format(task, avg_acc))


def run_nli_w_path(run_name, step_name, model_path):
    #run_name
    disable_eager_execution()
    hp = HPBert()
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    data_loader = nli.DataLoader(hp.seq_max, "bert_voca.txt", True)
    data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
    run_name = "{}_{}_NLI".format(run_name, step_name)
    saved_model = train_nli(hp, nli_setting, run_name, 3, data, model_path)
    tf.reset_default_graph()
    avg_acc = test_nli(hp, nli_setting, run_name, data, saved_model)
    print("avg_acc: ", avg_acc)

    save_report("nli", run_name, step_name, avg_acc)


def run_nli(run_name, step_name):
    model_path = get_model_path(run_name, step_name)
    return run_nli_w_path(run_name, step_name, model_path)

