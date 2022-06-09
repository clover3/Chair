from data_generator import tokenizer_wo_tf as tokenization
from tf_util.tf_logging import tf_logging
from tlm.token_utils import float_aware_strize


def log_print_inst(instance, features):
    tf_logging.info("*** Example ***")
    tf_logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in instance.tokens]))

    log_print_feature(features)


def log_print_feature(features):
    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
            values = feature.int64_list.value
        elif feature.float_list.value:
            values = feature.float_list.value
        tf_logging.info(
            "%s: %s" % (feature_name, " ".join([float_aware_strize(x) for x in values])))