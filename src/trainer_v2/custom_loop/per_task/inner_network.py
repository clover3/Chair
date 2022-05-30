from abc import ABC, abstractmethod

from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_checkpoint, load_stock_weights, \
    do_model_sanity_check
from trainer_v2.custom_loop.neural_network_def.assym_debug import BERT_AssymetricDebug
from trainer_v2.custom_loop.neural_network_def.assymetric import BERTAssymetric, BERTAsymmetricProjectMean
from trainer_v2.custom_loop.neural_network_def.siamese import BERTSiamese, BERTSiameseMean, BERTSiameseL


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
        do_model_sanity_check(self.network.bert_cls.l_bert)
        load_bert_checkpoint(self.network.bert_cls, init_checkpoint)
        do_model_sanity_check(self.network.bert_cls.l_bert)

    def do_sanity_check(self, msg):
        do_model_sanity_check(self.network.bert_cls.l_bert, msg)


class SiameseL(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        network = BERTSiameseL(bert_params, model_config)
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


class AsymmetricMeanPool(ClassificationModelIF):
    def __init__(self, inner_network=BERTAsymmetricProjectMean):
        super(AsymmetricMeanPool, self).__init__()
        self.inner_network = inner_network

    def build_model(self, bert_params, model_config):
        network = self.inner_network(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.network.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


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
