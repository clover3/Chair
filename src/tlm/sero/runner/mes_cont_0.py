from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.model.mes_cont import MES_continuous_0
from tlm.sero.mes_main_common import run_mes_variant
from tlm.training.train_flags import *


@report_run
def main(_):
    return run_mes_variant(MES_continuous_0)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()

