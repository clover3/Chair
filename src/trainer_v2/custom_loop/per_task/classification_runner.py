from trainer_v2.custom_loop.RunConfig2 import RunConfig2
from trainer_v2.custom_loop.modeling_common.bert_common import BertClassifier, load_bert_checkpoint
from trainer_v2.custom_loop.per_task.classification_inner_models import ClassificationInnerModelIF
from trainer_v2.custom_loop.per_task.classification_runner_factory import ClassificationRunnerFactory


class ClassificationClassic(ClassificationInnerModelIF):
    def build_model(self, bert_params, model_config):
        bert_classifier = BertClassifier(bert_params, model_config)
        self.bert_classifier = bert_classifier

    def get_keras_model(self):
        return self.bert_classifier.model

    def init_checkpoint(self, init_checkpoint):
        return load_bert_checkpoint(self.bert_classifier.bert_cls, init_checkpoint)


def get_classification_runner(bert_params, model_config, run_config: RunConfig2):
    inner = ClassificationClassic()
    return ClassificationRunnerFactory(bert_params, model_config, run_config, inner)


