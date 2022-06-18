import json
import os
from typing import Dict


def configurations():
    lr_l = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_size_l = [8, 16, 32, 64]
    epoch_l = [1, 2, 3, 4]
    d = {}
    for lr in lr_l:
        d['learning_rate'] = lr
        for batch_size in batch_size_l:
            d['batch_size'] = batch_size
            for epoch in epoch_l:
                d['train_epochs'] = epoch
                yield dict(d)


def configurations2():
    lr_l = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_size_l = [64, 128, 256, 512]
    epoch_l = [1, 2, 3, 4]
    d = {}
    for lr in lr_l:
        d['learning_rate'] = lr
        for batch_size in batch_size_l:
            d['batch_size'] = batch_size
            for epoch in epoch_l:
                d['train_epochs'] = epoch
                yield dict(d)


def configurations3():
    lr_l = [1e-5, 5e-6, 2e-6, 1e-6]
    batch_size_l = [256, 512, 1024, 2048]
    epoch_l = [2, 3, 4, 5, 6, 8]
    d = {}
    for lr in lr_l:
        d['learning_rate'] = lr
        for batch_size in batch_size_l:
            d['batch_size'] = batch_size
            for epoch in epoch_l:
                d['train_epochs'] = epoch
                yield dict(d)


def main():
    data_size = 392702
    config_list = list(configurations3())
    n_config = len(config_list)
    # random.seed(0)
    todo = list(range(n_config))
    # random.shuffle(todo)
    # print(todo)
    base_name = "ts44"
    for job_i in todo:
        config: Dict = config_list[job_i]
        steps_per_epoch = int(data_size / config['batch_size'])
        eval_every_n_step = steps_per_epoch
        train_step = steps_per_epoch * config['train_epochs']
        config['train_step'] = train_step
        config['eval_every_n_step'] = eval_every_n_step
        config['steps_per_epoch'] = steps_per_epoch
        steps_per_execution = int(steps_per_epoch / 10)
        config['steps_per_execution'] = steps_per_execution
        config['save_every_n_step'] = train_step
        config['eval_batch_size'] = 16
        run_name: str = f"ts43_{job_i}"
        # config["run_name"] = run_name
        save_path = os.path.join("data", "config_t", f"{base_name}_{job_i}")
        print(config)
        json.dump(config, open(save_path, "w"))


if __name__ == "__main__":
    main()