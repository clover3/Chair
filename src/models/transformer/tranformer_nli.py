from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import bert_get_hidden
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module


METHOD_CROSSENT = 2
METHOD_HINGE = 7

class transformer_nli:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [METHOD_CROSSENT, METHOD_HINGE]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)
        if method == 0:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.nn.sigmoid(cl)
            # cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            # self.pkc = self.conf_logits * self.rf_mask
            # rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
            rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
        elif method == 1:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf_module.cossim(cl, self.rf_mask)
            #self.pkc = self.conf_logits * self.rf_mask
            rl_loss_list = tf.reduce_sum(self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
        elif method == METHOD_CROSSENT:
            cl = tf.layers.dense(self.model.get_sequence_output(), 2, name="aux_conflict")
            probs = tf.nn.softmax(cl)
            losses = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.rf_mask, 2), logits=cl)
            self.conf_logits = probs[:,:,1] - 0.5
            self.rl_loss = tf.reduce_mean(losses)
        elif method == 3:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            self.bias = tf.Variable(0.0)
            self.conf_logits = (cl + self.bias)
            rl_loss_list = tf.nn.relu(1 - self.conf_logits * self.rf_mask)
            rl_loss_list = tf.reduce_mean(rl_loss_list, axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.losses.hinge_loss(labels, self.conf_logits)
            self.hinge_loss = tf.reduce_sum(hinge_losses)
        elif method == 4:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.losses.hinge_loss(labels, self.conf_logits)
            self.rl_loss = hinge_losses
        elif method == 5:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            self.labels = tf.cast(tf.greater(self.rf_mask, 0), tf.float32)
            self.rl_loss = tf.reduce_mean(tf_module.correlation_coefficient_loss(cl, -self.rf_mask))
        elif method == 6:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            #cl = tf.layers.dense(cl1, 1, name="aux_conflict2")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.nn.sigmoid(cl)
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf.reduce_sum(self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(tf_module.correlation_coefficient_loss(cl, -self.rf_mask))
        elif method == METHOD_HINGE:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            self.conf_logits = cl
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.losses.hinge_loss(labels, self.conf_logits)
            self.rl_loss = tf.reduce_sum(hinge_losses)

        self.conf_softmax = tf.nn.softmax(self.conf_logits, axis=-1)
#            self.rl_loss = tf.reduce_mean(rl_loss_list)
            #with tf.device("/device:GPU:1"):
            #    pl = tf.layers.dense(self.model.get_sequence_output(), hp.hidden_units, name="aux_pairing1")
            #    pl = tf.layers.dense(pl, 1, name="aux_pairing2")
            #    pl = tf.reshape(pl, [-1, seq_length])
            #    pl = tf.contrib.layers.layer_norm(pl)
            #    self.pair_logits = pl
            #    self.pr_mask = tf.placeholder(tf.float32, [None, seq_length])
            #    labels = tf.greater(self.pr_mask, 0)
            #    #hinge_losses = tf.losses.hinge_loss(labels, self.pair_logits)
            #    pr_loss_list = tf.reduce_sum(self.pair_logits * self.pr_mask, axis=1)
            #    #pr_loss_list = hinge_losses
            #    self.pr_loss = tf.reduce_mean(pr_loss_list)



class transformer_nli_hidden:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [METHOD_CROSSENT, METHOD_HINGE]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss

        self.embedding_output = self.model.get_embedding_output()
        self.all_layers = self.model.get_all_encoder_layers()


class transformer_nli_grad:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [METHOD_CROSSENT, METHOD_HINGE]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert_get_hidden.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss

        all_layer_grads = []
        all_layers = self.model.all_layer_outputs
        for i in range(len(all_layers)):
            grad = tf.gradients(self.logits, all_layers[i])
            all_layer_grads.append(grad)

        grad_emb = tf.gradients(self.logits, self.model.embedding_output)
        self.all_layer_grads = all_layer_grads
        self.grad_emb = grad_emb

class transformer_nli_embedding_in:
    def __init__(self, hp, voca_size, is_training):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
#        self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids
        self.encoded_embedding_in = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        self.attention_mask_in = tf.placeholder(tf.float32, [None, seq_length, seq_length])
        use_one_hot_embeddings = use_tpu
        self.model = bert.BertEmbeddingInOut(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            embeddding_as_input=(self.encoded_embedding_in, self.attention_mask_in),
        )

        self.encoded_embedding_out = self.model.embedding_output
        self.attention_mask_out = self.model.attention_mask

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

        cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
        cl = tf.reshape(cl, [-1, seq_length])
        #cl = tf.nn.sigmoid(cl)
        #cl = tf.contrib.layers.layer_norm(cl)
        self.conf_logits = cl
        #self.pkc = self.conf_logits * self.rf_mask
        #rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
        rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)

        num_tagged = tf.nn.relu(self.conf_logits+1)
        self.verbose_loss = tf.reduce_mean(tf.reduce_sum(num_tagged, axis=1))
        self.rl_loss = tf.reduce_mean(rl_loss_list)



class transformer_nli_vector:
    def __init__(self, hp, voca_size, is_training):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
#        self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        self.fixed_embedding = tf.get_variable("fixed_v", [hp.num_v, hp.fixed_v, hp.hidden_units], dtype=tf.float32,
                                               initializer=bert.create_initializer(config.initializer_range))
        self.encoded_embedding_in = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])

        batch_dyn = tf.shape(self.encoded_embedding_in)[0]
        tile_fixed_emb = tf.reshape(tf.tile(self.fixed_embedding, [batch_dyn, 1, 1]),
                                    [-1, hp.num_v, hp.fixed_v, hp.hidden_units])

        valid_input_embedding = self.encoded_embedding_in[:, hp.fixed_v:, :]
        tile_enc_emb = tf.reshape(tf.tile(valid_input_embedding, [hp.num_v,1,1]), [hp.num_v, -1, hp.seq_max - hp.fixed_v, hp.hidden_units])
        tile_enc_emb = tf.transpose(tile_enc_emb, [1, 0, 2, 3])
        concat_embedding = tf.concat([tile_enc_emb, tile_fixed_emb], 2)[:,:,:seq_length,:]
        concat_emb_flat = tf.reshape(concat_embedding, [-1, hp.seq_max, hp.hidden_units])
        self.attention_mask_in = tf.placeholder(tf.float32, [None, seq_length, seq_length])

        def repeat_num_v(t):
            tile_param = [hp.num_v] + tf.shape(t)[1:]
            t = tf.tile(t, tile_param)
            last_shape = [-1] + tf.shape(t)[1:]
            return tf.reshape(t, last_shape)


        # If we directly feed input_ids, it will get locations embedding of begging, while we need input_ids to be second segments.
        # We bypass this by first retrieving word embedding only.
        attention_mask_repeat = tf.reshape(tf.tile(self.attention_mask_in, [hp.num_v,1,1]), [-1, seq_length, seq_length])

        def repeat_dummy(in_tensor):
            return tf.concat([in_tensor[:, :hp.fixed_v], in_tensor[:, :-hp.fixed_v]], 1)

        input_ids_pad = repeat_dummy(input_ids)
        input_mask_pad = repeat_dummy(input_mask)
        segment_ids_pad = repeat_dummy(segment_ids)

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertEmbeddingInOut(
            config=config,
            is_training=is_training,
            input_ids=input_ids_pad,
            input_mask=input_mask_pad,
            token_type_ids=segment_ids_pad,
            use_one_hot_embeddings=use_one_hot_embeddings,
            embeddding_as_input=(concat_emb_flat, attention_mask_repeat),
        )

        self.encoded_embedding_out = self.model.embedding_output
        self.attention_mask_out = self.model.attention_mask

        def predict(enc, Y, is_train):
            if is_train:
                mode = tf.estimator.ModeKeys.TRAIN
            else:
                mode = tf.estimator.ModeKeys.EVAL
            return predict_ex(enc, Y, mode)

        def predict_ex(enc, Y, mode):
            feature_loc = 0
            logits_raw = tf.layers.dense(enc[:, feature_loc, :], nli.num_classes, name="cls_dense")
            if hp.use_reorder :
                logits_reorder = [logits_raw[:,1], logits_raw[:,0], logits_raw[:,2]]
                logits_candidate = tf.stack(logits_reorder, axis=1)  # [-1, 3]
            else:
                logits_candidate = logits_raw
            logits_candidate = tf.reshape(logits_candidate, [-1, hp.num_v, nli.num_classes])
            soft_candidate = tf.nn.softmax(logits_candidate)
            active_arg = tf.cast(tf.argmin(soft_candidate[:,:,0], axis=1), dtype=tf.int32) # [batch]
            indice = tf.stack([tf.range(batch_dyn), active_arg], axis=1)
            print(indice.shape)
            logits = tf.gather_nd(logits_candidate, indice)
            print(logits_candidate.shape)
            print(logits.shape)

            labels = tf.one_hot(Y, nli.num_classes)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            self.acc = tf_module.accuracy(logits, Y)
            self.logits = logits
            tf.summary.scalar("acc", self.acc)
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                self.loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits,
                    labels=labels)

                neg = tf.cast(tf.equal(preds, 0), tf.float32) * tf.constant(0.1)
                pos = tf.cast(tf.not_equal(preds, 0), tf.float32)

                weight_losses = self.loss_arr * (pos + neg)
                # TP : 1
                # FN : 0.1
                # FP : 1
                # TN : 0.1
                self.loss = tf.reduce_mean(weight_losses)
                tf.summary.scalar("loss", self.loss)
                return preds, self.loss
            else:
                return preds

        pred, loss = predict(self.model.get_sequence_output(), label_ids, True)
        #pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

        cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
        cl = tf.reshape(cl, [-1, seq_length])
        #cl = tf.nn.sigmoid(cl)
        #cl = tf.contrib.layers.layer_norm(cl)
        self.conf_logits = cl
        #self.pkc = self.conf_logits * self.rf_mask
        #rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
        rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)

        num_tagged = tf.nn.relu(self.conf_logits+1)
        self.verbose_loss = tf.reduce_mean(tf.reduce_sum(num_tagged, axis=1))
        self.rl_loss = tf.reduce_mean(rl_loss_list)





