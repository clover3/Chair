import argparse
import sys

import numpy as np
import tensorflow as tf

from attribution.eval import predict_translate
from cache import save_to_pickle
from data_generator.NLI.nli import get_modified_data_loader2, tags
from data_generator.shared_setting import BertNLI
from explain.explain_trainer import ExplainPredictor
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import load_model
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session

parser = argparse.ArgumentParser(description='')
parser.add_argument("--tag", help="Your input file.")
parser.add_argument("--model_path", help="Your model path.")
parser.add_argument("--run_name", )
parser.add_argument("--data_id")
parser.add_argument("--modeling_option")


def predict_nli_ex(hparam, nli_setting, data_loader,
                   explain_tag, data_id, model_path, run_name, modeling_option):
    print("predict_nli_ex")
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    batches = get_batches_ex(enc_payload, hparam.batch_size, 3)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)

    tag_idx = tags.index(explain_tag)
    explain_predictor = ExplainPredictor(len(tags), task.model.get_sequence_output(), modeling_option)
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    load_model(sess, model_path)

    ex_logits_list = []
    for batch in batches:
        x0, x1, x2 = batch
        ex_logits, = sess.run([explain_predictor.get_score()[tag_idx]],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                           })
        ex_logits_list.append(ex_logits)
    ex_logits = np.concatenate(ex_logits_list)
    pred_list = predict_translate(ex_logits, data_loader, enc_payload, plain_payload)
    save_to_pickle(pred_list, "pred_{}_{}".format(run_name, data_id))


def run(args):
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp, nli_setting)

    predict_nli_ex(hp, nli_setting, data_loader,
                         args.tag,
                         args.data_id,
                         args.model_path,
                         args.run_name,
                         args.modeling_option,
                         )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)