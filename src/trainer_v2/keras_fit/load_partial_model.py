from trainer_v2.chair_logging import c_log
from trainer_v2.keras_fit.runner.siamese import get_model_config
from trainer_v2.keras_fit.siamese_model import model_factory_siamese
from trainer_v2.run_config import RunConfigEx


def partial_init(init_checkpoint, sub_model):
    c_log.info("Loading model from {}".format(init_checkpoint))

def main():
    model_config = get_model_config()
    run_config = RunConfigEx(train_step=100,
                             steps_per_execution=1,
                             steps_per_epoch=10,
                             )
    get_model_fn = model_factory_siamese(model_config, run_config)

    model, inner_model_list = get_model_fn()
    model.summary()
    sub_model = inner_model_list[0]
    var = [v for v in sub_model.trainable_variables]
    for v in var:
        print(v.name)



if __name__ == "__main__":
    main()