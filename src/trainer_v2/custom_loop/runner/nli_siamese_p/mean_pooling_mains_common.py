from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.inner_network import SiameseMeanProject
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop import tf_run_for_bert


def mean_pooling_common(args, project_dim):
    c_log.info("Main classification-siamese mean pooling")
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()
    model_config.project_dim = project_dim
    network = SiameseMeanProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    tf_run_for_bert(dataset_factory, model_config, run_config, network)