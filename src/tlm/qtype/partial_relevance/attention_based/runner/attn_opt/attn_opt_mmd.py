import os
from typing import List

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_v2_support import disable_eager_execution
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt import \
    TransformerAttentionOptModel
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt_wrap import \
    AttentionMaskOptimizer, AttnOptEncoderWrap, init_model_for_inference, inference
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import AttnOptHP
from contradiction.alignment.data_structure.print_helper import rei_to_text
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


class MMDHP(AttnOptHP):
    num_classes = 2
    lr = 1e-2
    init_log_alpha = 0
    factor = 0.01


def get_attention_mask_optimizer_mmd_z(hp):
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    model = TransformerAttentionOptModel(hp)
    optimizer = AttentionMaskOptimizer(model, hp)
    optimizer.load_model(save_path)
    return optimizer


def mmd_item():
    problems: List[RelatedEvalInstance] = load_mmde_problem("dev_sent")
    tokenizer = get_tokenizer()
    for p in problems:
        print(rei_to_text(tokenizer, p))
        yield p.seg_instance


def main():
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    disable_eager_execution()
    hp = MMDHP()
    print(hp.factor)
    encoder = AttnOptEncoderWrap(hp.seq_max)
    optimizer: AttentionMaskOptimizer = get_attention_mask_optimizer_mmd_z(hp)
    num_steps = 100

    for inst in mmd_item():
        item = encoder.encode(inst)
        optimizer.train(item, num_steps)
        fetch_inf_mask = optimizer.fetch_inf_mask(item)
        item["given_mask"] = fetch_inf_mask
        print("Getting infernece")
        task, sess = init_model_for_inference(hp, save_path)
        inference(sess, task, item)


if __name__ == "__main__":
    main()
