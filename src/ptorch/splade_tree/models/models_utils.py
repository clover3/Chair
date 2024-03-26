from omegaconf import DictConfig

from .cross_encoder_like import CrossEncoderLikeBase, TransformerWeightedSum
from ..models.transformer_rep import Splade, SpladeDoc


def get_model(config: DictConfig, init_dict: DictConfig):
    # no need to reload model here, it will be done later
    # (either in train.py or in Evaluator.__init__()
    matching_type = config["matching_type"]
    model_map = {
        "splade": Splade,
        "splade_doc": SpladeDoc,
        "cross_encoder": CrossEncoderLikeBase,
        "pep": TransformerWeightedSum,
    }
    try:
        model_class = model_map[matching_type]
    except KeyError:
        raise NotImplementedError("provide valid matching type ({})".format(matching_type))
    return model_class(**init_dict)


def print_a():
    print("A")