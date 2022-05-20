from abc import ABC, abstractmethod

from trainer_v2.custom_loop.modeling_common.assym_debug import BERT_AssymetricDebug
from trainer_v2.custom_loop.modeling_common.assymetric import BERT_Assymetric
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_checkpoint


class ClassificationInnerModelIF(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build_model(self, bert_params, model_config):
        pass

    @abstractmethod
    def get_keras_model(self):
        pass

    @abstractmethod
    def init_checkpoint(self, init_checkpoint):
        pass


class ClassificationAsym(ClassificationInnerModelIF):
    def build_model(self, bert_params, model_config):
        classifier = BERT_Assymetric(bert_params, model_config)
        self._classifier = classifier

    def get_keras_model(self):
        return self._classifier.model

    def init_checkpoint(self, init_checkpoint):
        bert_cls1, bert_cls2 = self._classifier.bert_cls_list
        load_bert_checkpoint(bert_cls1, init_checkpoint)
        load_bert_checkpoint(bert_cls2, init_checkpoint)


class ClassificationAsymDebug(ClassificationInnerModelIF):
    def build_model(self, bert_params, model_config):
        classifier = BERT_AssymetricDebug(bert_params, model_config)
        self._classifier = classifier

    def get_keras_model(self):
        return self._classifier.model

    def init_checkpoint(self, init_checkpoint):
        bert_cls1, bert_cls2 = self._classifier.bert_cls_list
        load_bert_checkpoint(bert_cls1, init_checkpoint)
        load_bert_checkpoint(bert_cls2, init_checkpoint)