import sys

from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel


def main():
    src_path = sys.argv[1]
    model_save_path = sys.argv[2]
    model_config = ModelConfig512_1()
    task_model = TwoSegConcatLogitCombineTwoModel(model_config, CombineByScoreAdd)
    task_model.build_model(None)
    task_model.load_checkpoint(src_path)
    task_model.point_model.save(model_save_path)




if __name__ == "__main__":
    main()