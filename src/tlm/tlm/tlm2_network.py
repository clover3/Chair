from models.transformer import bert_common_v2 as bert_common
from models.transformer import hyperparams
from tlm.model.base import BertModel
from tlm.tlm.loss_diff_common import IndependentLossModel


def tlm2(bert_config, use_one_hot_embeddings, features):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    hp = hyperparams.HPBert()
    voca_size = 30522
    sequence_shape = bert_common.get_shape_list2(input_ids)

    encode_model = BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    loss_model = IndependentLossModel(bert_config)
    loss_model.build_predictions(encode_model.get_sequence_output())
    output = -(loss_model.prob1 - loss_model.prob2)
    return output


def tlm_prefer_hard(bert_config, use_one_hot_embeddings, features):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    encode_model = BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    loss_model = IndependentLossModel(bert_config)
    loss_model.build_predictions(encode_model.get_sequence_output())
    # if score is higher, it is more often sampled
    output = -loss_model.prob1
    return output




def tlm2_raw_prob(bert_config, use_one_hot_embeddings, input_ids, input_mask, segment_ids):
    encode_model = BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    loss_model = IndependentLossModel(bert_config)
    loss_model.build_predictions(encode_model.get_sequence_output())
    output = -(loss_model.prob1 - loss_model.prob2)
    return output, loss_model.prob1, loss_model.prob2
