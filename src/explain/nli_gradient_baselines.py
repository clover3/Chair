import tensorflow as tf

from attribution.eval import predict_translate
from cache import save_to_pickle
from models.transformer.nli_base import transformer_nli_pooled_embedding_in
from trainer.model_saver import load_model
from trainer.tf_train_module import init_session


def nli_attribution_predict(hparam, nli_setting, data_loader,
                            explain_tag, method_name, data_id, sub_range, model_path):
    enc_payload, plain_payload = data_loader.load_plain_text(data_id)
    if sub_range is not None:
        raise Exception("Sub_range is not supported")


    from attribution.gradient import explain_by_gradient
    from attribution.deepexplain.tensorflow import DeepExplain

    sess = init_session()

    with DeepExplain(session=sess, graph=sess.graph) as de:
        task = transformer_nli_pooled_embedding_in(hparam, nli_setting.vocab_size, False)
        softmax_out = tf.nn.softmax(task.logits, axis=-1)
        sess.run(tf.global_variables_initializer())
        load_model(sess, model_path)
        emb_outputs = task.encoded_embedding_out, task.attention_mask_out
        emb_input = task.encoded_embedding_in, task.attention_mask_in

        def feed_end_input(batch):
            x0, x1, x2 = batch
            return {task.x_list[0]:x0,
                    task.x_list[1]:x1,
                    task.x_list[2]:x2,
                    }

        explains = explain_by_gradient(enc_payload, method_name, explain_tag, sess, de,
                                       feed_end_input, emb_outputs, emb_input, softmax_out)

        pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
        save_to_pickle(pred_list, "pred_{}_{}".format(method_name, data_id))
