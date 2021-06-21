from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.model.mes_hinge import MES_hinge_layer1_load_many_loss, MES_pred_layer2_load
from tlm.sero.mes_pair_common import run_mes_pairwise
from tlm.training.train_flags import *


@report_run
def main(_):
    return run_mes_pairwise(MES_hinge_layer1_load_many_loss, MES_pred_layer2_load)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
