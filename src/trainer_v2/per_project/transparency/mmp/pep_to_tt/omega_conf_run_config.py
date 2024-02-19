from trainer_v2.custom_loop.run_config2 import CommonRunConfig, TrainConfig, DeviceConfig, DatasetConfig, RunConfig2, \
    EvalConfig





def get_run_config_from_omega_conf(omega_conf):
    common_run_config = CommonRunConfig(batch_size=omega_conf.batch_size)

    train_config = TrainConfig(
        train_step=omega_conf.train_step,
        save_every_n_step=omega_conf.save_every_n_step,
        eval_every_n_step=omega_conf.eval_every_n_step,
        init_checkpoint=omega_conf.init_checkpoint,
        model_save_path=omega_conf.model_save_path
    )
    device_config = DeviceConfig()
    dataset_config = DatasetConfig(omega_conf.train_data_dir, omega_conf.eval_data_dir)
    run_config = RunConfig2(common_run_config=common_run_config,
                            dataset_config=dataset_config,
                            train_config=train_config,
                            device_config=device_config
                            )

    return run_config


def get_eval_run_config_from_omega_conf(omega_conf):
    common_run_config = CommonRunConfig(
        batch_size=omega_conf.batch_size,
        run_name=omega_conf.run_name
    )
    eval_config = EvalConfig(
        model_save_path=omega_conf.model_save_path,
        eval_step=omega_conf.eval_steps,
    )
    device_config = DeviceConfig()
    dataset_config = DatasetConfig(omega_conf.train_data_dir, omega_conf.eval_data_dir)
    run_config = RunConfig2(common_run_config=common_run_config,
                            dataset_config=dataset_config,
                            eval_config=eval_config,
                            device_config=device_config
                            )

    return run_config
