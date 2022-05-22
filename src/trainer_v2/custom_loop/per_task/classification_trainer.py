from trainer_v2.custom_loop.modeling_common.bert_common import BertClassifier, load_bert_checkpoint
from trainer_v2.custom_loop.per_task.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import RunConfig2


class StandardBertCls(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        bert_classifier = BertClassifier(bert_params, model_config)
        self.bert_classifier = bert_classifier

    def get_keras_model(self):
        return self.bert_classifier.model

    def init_checkpoint(self, init_checkpoint):
        return load_bert_checkpoint(self.bert_classifier.bert_cls, init_checkpoint)


def get_classification_trainer(bert_params, model_config, run_config: RunConfig2):
    inner = StandardBertCls()
    return Trainer(bert_params, model_config, run_config, inner)


