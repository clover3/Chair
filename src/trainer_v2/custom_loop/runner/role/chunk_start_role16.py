from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.role_aug import ChunkStartRole
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerNoSum
from trainer_v2.custom_loop.runner.nli_asymmetric.two_seg_commons import two_seg_common2

if __name__ == "__main__":
    c_log.info("Start {}".format(__file__))
    hidden_dim = 16

    def get_lower_project_layer(out_dimension):
        return TwoLayerDense(hidden_dim,
                             out_dimension,
                             activation1='relu',
                             activation2='relu', )

    inner = ChunkStartRole(FuzzyLogicLayerNoSum, get_lower_project_layer)
    two_seg_common2(inner)
