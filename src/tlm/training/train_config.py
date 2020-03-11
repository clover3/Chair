class TrainConfigEx:
    def __init__(self,
                 init_checkpoint,  # 0th
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,  # 5th
                 num_classes,
                 max_predictions_per_seq,
                 gradient_accumulation=1,
                 checkpoint_type="",
                 second_init_checkpoint="",  # 10th
                 use_old_logits=False,
                 learning_rate2=0,
                 fixed_mask=False,
                 random_seed=None,
                 no_lr_decay=False,  # 15th
                 ):
        self.init_checkpoint = init_checkpoint  # 0th
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings  # 5th
        self.num_classes = num_classes
        self.max_predictions_per_seq = max_predictions_per_seq
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_type = checkpoint_type
        self.second_init_checkpoint = second_init_checkpoint  # 10th
        self.use_old_logits = use_old_logits
        self.learning_rate2 = learning_rate2
        self.fixed_mask = fixed_mask
        self.random_seed = random_seed
        self.no_lr_decay = no_lr_decay  # 15th

    @classmethod
    def from_flags(cls, flags):
        return TrainConfigEx(
            flags.init_checkpoint,  # 0th
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,  # 5th
            flags.num_classes,
            flags.max_predictions_per_seq,
            flags.gradient_accumulation,
            flags.checkpoint_type,
            flags.target_task_checkpoint,  # 10th
            flags.use_old_logits,
            flags.learning_rate2,
            flags.fixed_mask,
            flags.random_seed,
            flags.no_lr_decay  # 15th
        )
