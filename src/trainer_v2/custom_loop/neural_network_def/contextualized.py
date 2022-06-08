import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.transformer import TransformerEncoderLayer
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, load_stock_weights, \
    load_stock_weights_encoder_only
from trainer_v2.custom_loop.modeling_common.network_utils import MeanProjectionEnc, MeanProjection, SplitSegmentIDLayer, \
    TileAfterExpandDims, VectorThreeFeature, TwoLayerDense, SplitSegmentIDLayerWVar, SplitSegmentIDWMeanLayer
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF


class BERTAsymmetricContextualizedSlice(ClassificationModelIF):
    def __init__(self, decision_combine_layer):
        super(BERTAsymmetricContextualizedSlice, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2_upper = TransformerEncoderLayer.from_params(
            bert_params,
            name="encoder2_b/encoder"
        )
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, l_token_type_ids2], training=False)

        ssi = SplitSegmentIDLayer()
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_token_type_ids2))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_seq_flat = encoder2_upper(rep_middle_concat)
        rep2_flat = mp(rep2_seq_flat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)
        # hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        # local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]
        self.l_bert_like = encoder2_upper

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)

        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_bert_like, init_checkpoint, n_expected_restore=197 - n_embedding_vars)


class BERTAsymmetricContextualizedSlice2(ClassificationModelIF):
    def __init__(self, decision_combine_layer):
        super(BERTAsymmetricContextualizedSlice2, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2_upper = TransformerEncoderLayer.from_params(
            bert_params,
            name="encoder2_b/encoder"
        )
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        l_slice_ids = l_token_type_ids2
        dummy_l_token_type_ids = tf.zeros_like(l_token_type_ids2)

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, dummy_l_token_type_ids], training=False)

        ssi = SplitSegmentIDLayer()
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_slice_ids))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_seq_flat = encoder2_upper(rep_middle_concat)
        rep2_flat = mp(rep2_seq_flat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)

        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]
        self.l_bert_like = encoder2_upper

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)

        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_bert_like, init_checkpoint, n_expected_restore=197 - n_embedding_vars)


class BERTAsymmetricContextualizedSlice3(ClassificationModelIF):
    def __init__(self, decision_combine_layer):
        super(BERTAsymmetricContextualizedSlice3, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        l_slice_ids = l_token_type_ids2
        dummy_l_token_type_ids = tf.zeros_like(l_token_type_ids2)

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, dummy_l_token_type_ids], training=False)

        ssi = SplitSegmentIDLayer()
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_slice_ids))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_flat = mp(rep_middle_concat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)

        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)


class BERTAsymmetricContextualizedSlice4(ClassificationModelIF):
    # No upper encoder
    def __init__(self, decision_combine_layer):
        super(BERTAsymmetricContextualizedSlice4, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        l_slice_ids = l_token_type_ids2
        dummy_l_token_type_ids = tf.zeros_like(l_token_type_ids2)

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, dummy_l_token_type_ids], training=False)

        ssi = SplitSegmentIDLayerWVar(bert_params.hidden_size)
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_slice_ids))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_flat = mp(rep_middle_concat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)

        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)


class BERTAsymmetricContextualizedSlice5(ClassificationModelIF):
    def __init__(self, decision_combine_layer):
        super(BERTAsymmetricContextualizedSlice5, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2_upper = TransformerEncoderLayer.from_params(
            bert_params,
            name="encoder2_b/encoder"
        )
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, l_token_type_ids2], training=True)

        ssi = SplitSegmentIDLayer()
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_token_type_ids2))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_seq_flat = encoder2_upper(rep_middle_concat)
        rep2_flat = mp(rep2_seq_flat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="two_towers")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]
        self.l_bert_like = encoder2_upper

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)

        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_bert_like, init_checkpoint, n_expected_restore=197 - n_embedding_vars)


class BAContextualizedSlice6(ClassificationModelIF):
    # No upper encoder
    # Trainable lowe encoder
    def __init__(self, decision_combine_layer):
        super(BAContextualizedSlice6, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = True
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        l_slice_ids = l_token_type_ids2
        dummy_l_token_type_ids = tf.zeros_like(l_token_type_ids2)

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_middle = encoder2_lower([l_input_ids2, dummy_l_token_type_ids], training=True)

        ssi = SplitSegmentIDLayerWVar(bert_params.hidden_size)
        rep_middle0, rep_middle1 = ssi((rep2_middle, l_input_ids2, l_slice_ids))
        rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
        rep2_flat = mp(rep_middle_concat)

        batch_size, _ = get_shape_list2(l_input_ids2)
        _, rep_dim = get_shape_list2(rep2_flat)
        rep2_stacked = tf.reshape(rep2_flat, [batch_size, num_window, rep_dim])

        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)

        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)


class BAContextualizedSlice7(ClassificationModelIF):
    # No upper encoder
    # Trainable lowe encoder
    def __init__(self, decision_combine_layer):
        super(BAContextualizedSlice7, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        mp = MeanProjection(config.project_dim, "encoder2_b")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2"))
        encoder2_lower.trainable = True
        num_classes = config.num_classes
        num_window = 2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        l_slice_ids = l_token_type_ids2
        dummy_l_token_type_ids = tf.zeros_like(l_token_type_ids2)

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        rep2_all = encoder2_lower([l_input_ids2, dummy_l_token_type_ids], training=True)
        self.projector = tf.keras.layers.Dense(config.project_dim, activation='relu', name="{}/project".format("encoder2"))
        rep2_projected = self.projector(rep2_all)

        ssi = SplitSegmentIDWMeanLayer()
        # Zero-masked
        rep_middle0, rep_middle1 = ssi((rep2_projected, l_input_ids2, l_slice_ids))
        rep2_stacked = tf.stack([rep_middle0, rep_middle1], axis=1)
        rep1_stacked = TileAfterExpandDims(1, [1, num_window, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes)(feature_rep)

        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)