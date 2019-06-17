from trainer.tf_module import get_batches_ex
import random
from misc_lib import pick1, sample_prob

class SharedFeeder:
    def __init__(self, data_loader_list, proportion_list, task_name_list, batch_size):
        self.batch_size = batch_size
        self.train_batch_list = []
        self.dev_batch_list = []
        self.prob_sum = sum(proportion_list)
        self.task_prob = list([p/self.prob_sum for p in proportion_list])
        self.task_name_list = task_name_list

        for task_i, data_loader in enumerate(data_loader_list):
            train_data = data_loader.get_train_data()
            dev_data = data_loader.get_dev_data()

            self.train_batch_list.append(get_batches_ex(train_data, batch_size, 4))
            self.dev_batch_list.append(get_batches_ex(dev_data, batch_size, 4))
            print("{} : {} train batches".format(self.task_name_list[task_i], len(self.train_batch_list[task_i])))
            print("{} : {} dev batches".format(self.task_name_list[task_i], len(self.dev_batch_list[task_i])))

    def sample_task(self):
        return sample_prob(enumerate(self.task_prob))

    def get_train_batch(self):
        return self.sample_batch(self.train_batch_list)

    def get_dev_batch(self):
        return self.sample_batch(self.dev_batch_list)

    def sample_batch(self, source_batch_list):
        task_idx = self.sample_task()
        batches = source_batch_list[task_idx]
        batch = pick1(batches)
        return task_idx, batch

    def get_dev_batch_from(self, task_idx):
        batches = self.dev_batch_list[task_idx]
        batch = pick1(batches)
        return task_idx, batch
