from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
# /home/youngwookim_umass_edu/work/miniconda3/envs/tf29/lib/python3.9/site-packages/transformers/optimization.py:306:
# FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version.
# Use the PyTorch implementation torch.optim.AdamW instead,
# or set `no_deprecation_warning=True` to disable this warning
def init_simple_bert_optim(model, lr, weight_decay, warmup_steps, num_training_steps):
    """
    inspired from https://github.com/ArthurCamara/bert-axioms/blob/master/scripts/bert.py
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler
