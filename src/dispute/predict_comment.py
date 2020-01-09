from cpath import model_path
from dispute.adreaction import FLAGS
from dispute.guardian import save_local_pickle, load_local_pickle
from models.cnn import CNN
from trainer.tf_module import *


def get_predictor():
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    cnn = CNN("agree",
              sequence_length=FLAGS.comment_length,
              num_classes=3,
              filter_sizes=[1, 2, 3],
              num_filters=64,
              init_emb=load_local_pickle("init_embedding"),
              embedding_size=FLAGS.embedding_size,
              dropout_prob=dropout_keep_prob
              )
    input_comment = tf.placeholder(tf.int32,
                                   shape=[None, FLAGS.comment_length],
                                   name="comment_input")
    #sout = model.cnn.network(input_comment)
    sout = cnn.network(input_comment)
    sess = init_session()
    batch_size = 512
    path = os.path.join(model_path, "runs", "agree", "model-36570")
    variables = tf.contrib.slim.get_variables_to_restore()
    for v in variables:
        print(v.name)
    loader = tf.train.Saver(variables)
    loader.restore(sess, path)
    def predict(comments):
        batches = get_batches_ex(comments, batch_size, 1)
        all_scores = []
        ticker = TimeEstimator(len(batches))
        for batch in batches:
            scores,  = sess.run([sout], feed_dict={
                input_comment:batch[0],
                dropout_keep_prob:1.0,
            })
            all_scores.append(scores)
            ticker.tick()

        return np.concatenate(all_scores)
    return predict


def predict_all_disagree():
    predict = get_predictor()
    discussion_list = load_local_pickle("code_comments")

    id_list = []
    all_comments = []
    for short_id, _,comments in discussion_list:
        id_list.append(short_id)
        new_format = []
        for i in range(len(comments)):
            new_format.append([comments[i]])
        all_comments.append(new_format)

    logit_list = flat_apply_stack(predict, all_comments, True)

    result = list(zip(id_list, logit_list))

    save_local_pickle(result, "disagreements")



if __name__ == "__main__":
    predict_all_disagree()
