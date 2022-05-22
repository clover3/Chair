from abc import ABC, abstractmethod

from trainer_v2.custom_loop.modeling_common.assym_debug import BERT_AssymetricDebug
from trainer_v2.custom_loop.modeling_common.assymetric import BERTAssymetric
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_checkpoint, load_stock_weights
from trainer_v2.custom_loop.modeling_common.siamese import BERTSiamese, BERTSiameseMean


class ClassificationModelIF(ABC):
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


class Siamese(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        network = BERTSiamese(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.network.bert_cls, init_checkpoint)


class Asymmetric(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        network = BERTAssymetric(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        bert_cls1, bert_cls2 = self.network.bert_cls_list
        load_bert_checkpoint(bert_cls1, init_checkpoint)
        load_bert_checkpoint(bert_cls2, init_checkpoint)


class AsymDebug(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        classifier = BERT_AssymetricDebug(bert_params, model_config)
        self.network = classifier

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        bert_cls1, bert_cls2 = self.network.bert_cls_list
        load_bert_checkpoint(bert_cls1, init_checkpoint)
        load_bert_checkpoint(bert_cls2, init_checkpoint)


class SiameseMeanProject(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        network = BERTSiameseMean(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        load_stock_weights(self.network.l_bert, init_checkpoint, n_expected_restore=197)
