class LMTrainConfig:
    def __init__(self,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,
                 max_predictions_per_seq,
                 gradient_accumulation=1,
                 checkpoint_type="",
                 second_init_checkpoint="",
                 fixed_mask=False,
                 random_seed=None,
                 no_lr_decay=False,
                 ):
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_predictions_per_seq = max_predictions_per_seq
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_type = checkpoint_type
        self.second_init_checkpoint = second_init_checkpoint
        self.fixed_mask = fixed_mask
        self.random_seed = random_seed
        self.no_lr_decay = no_lr_decay

    @classmethod
    def from_flags(cls, flags):
        return LMTrainConfig(
            flags.init_checkpoint,
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,
            flags.max_predictions_per_seq,
            flags.gradient_accumulation,
            flags.checkpoint_type,
            flags.target_task_checkpoint,
            flags.fixed_mask,
            flags.random_seed,
            flags.no_lr_decay
        )


class TrainConfig:
    def __init__(self,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,
                 num_classes,
                 gradient_accumulation,
                 iterations_per_loop,
                 checkpoint_type,
                 use_old_logits,
                 learning_rate2,
                 no_lr_decay,
                 ):
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.num_classes = num_classes
        self.gradient_accumulation = gradient_accumulation
        self.iterations_per_loop = iterations_per_loop
        self.checkpoint_type = checkpoint_type
        self.use_old_logits = use_old_logits
        self.learning_rate2 = learning_rate2
        self.no_lr_decay = no_lr_decay

    @classmethod
    def from_flags(cls, flags):
        return TrainConfig(
            flags.init_checkpoint,
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,
            flags.num_classes,
            flags.gradient_accumulation,
            flags.iterations_per_loop,
            flags.checkpoint_type,
            flags.use_old_logits,
            flags.learning_rate2,
            flags.no_lr_decay,
        )