from trainer_v2.custom_loop.modeling_common.bert_common import BertClassifier, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF


class StandardBertCls(BertBasedModelIF):
    def build_model(self, bert_params, model_config):
        bert_classifier = BertClassifier(bert_params, model_config)
        self.bert_classifier = bert_classifier

    def get_keras_model(self):
        return self.bert_classifier.model

    def init_checkpoint(self, init_checkpoint):
        return load_bert_checkpoint(self.bert_classifier.bert_cls, init_checkpoint)




