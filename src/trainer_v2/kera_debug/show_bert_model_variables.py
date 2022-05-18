from trainer_v2.partial_processing.assymetric_model import model_factory_assym
from trainer_v2.partial_processing.config_helper import get_bert_config
from trainer_v2.partial_processing.modeling import get_transformer_encoder
from trainer_v2.partial_processing.runner.two_seg_classifier import get_model_config
from trainer_v2.run_config import RunConfigEx



def main():
    bert_config = get_bert_config()
    bert_encoder = get_transformer_encoder(bert_config)
    for v in bert_encoder.variables:
        print(v.name)


def main2():
    model_config = get_model_config()
    run_config = RunConfigEx(train_step=19)
    get_model_fn = model_factory_assym(model_config, run_config)
    model, sub_model_list = get_model_fn()

    for m in sub_model_list:
        print(m)
        for v in m.variables:
            print(v.name)


if __name__ == "__main__":
    main2()