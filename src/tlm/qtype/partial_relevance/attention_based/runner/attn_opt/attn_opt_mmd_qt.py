import os

from cpath import output_path
from tf_v2_support import disable_eager_execution
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt_qt import \
    TransformerAttentionOptQTModel
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt_wrap import \
    AttentionMaskOptimizer, AttnOptQTEncoder
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import AttnOptHP
from alignment.data_structure.eval_data_structure import get_test_segment_instance


class MMDHP(AttnOptHP):
    num_classes = 2
    init_log_alpha = 0


def get_attention_mask_optimizer_mmd_z(hp):
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    model = TransformerAttentionOptQTModel(hp)
    optimizer = AttentionMaskOptimizer(model, hp)
    optimizer.load_model(save_path)
    return optimizer


def main():
    disable_eager_execution()
    inst = get_test_segment_instance()
    hp = MMDHP()
    encoder = AttnOptQTEncoder(hp.seq_max)
    optimizer: AttentionMaskOptimizer = get_attention_mask_optimizer_mmd_z(hp)
    print("query tokens", len(inst.text1.tokens_ids))
    print("doc tokens", len(inst.text2.tokens_ids))
    num_steps = 500
    item = encoder.encode(inst, 1)
    optimizer.train(item, num_steps)


if __name__ == "__main__":
    main()
