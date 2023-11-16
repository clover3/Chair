import  tensorflow as tf


from cpath import output_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log

from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeOnBERT


def main():
    model_path = path_join(output_path, "model", "runs", "mmp1")
    c_log.info("Loading model")
    ranking_model = tf.keras.models.load_model(model_path, compile=False)

    for l in ranking_model.layers:
        print(l.name)

    model = ProbeOnBERT(ranking_model)
    for sub_name, sub_features in model.probe_model_output.items():
        print(sub_name)
        for k, v in sub_features.items():
            print(k, v)


if __name__ == "__main__":
    main()
